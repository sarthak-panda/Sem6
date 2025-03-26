#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "modify.cuh"

using namespace std;

// Helper function to find the next power of two
int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

// Kernel to initialize frequency arrays
__global__ void initFreqKernel(int* freq_global, const int* d_range, int max_freq_size, int numMatrices) {
    int k = blockIdx.x;
    if (k >= numMatrices) return;
    int maxV = d_range[k];
    int* freqArray = &freq_global[k * max_freq_size];
    int tid = threadIdx.x;
    for (int i = tid; i <= maxV; i += blockDim.x) {
        freqArray[i] = 0;
    }
}

// Kernel to count frequencies using multiple blocks per matrix
__global__ void countFreqKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, const int* d_prefix_blocks, int numMatrices, int* freq_global, int max_freq_size, int threadsPerBlock) {
    int block_idx = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks[i] <= block_idx && block_idx < d_prefix_blocks[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;

    int maxV = d_range[matrix_k];
    int elements = d_rows[matrix_k] * d_cols[matrix_k];
    int* freqArray = &freq_global[matrix_k * max_freq_size];

    int offset = 0;
    for (int m = 0; m < matrix_k; m++) {
        offset += d_rows[m] * d_cols[m];
    }
    int* matrix = d_input + offset;

    int blocks_before = block_idx - d_prefix_blocks[matrix_k];
    int start = blocks_before * threadsPerBlock;
    int end = min(start + threadsPerBlock, elements);

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int val = matrix[i];
        if (val <= maxV) {
            atomicAdd(&freqArray[val], 1);
        }
    }
}

// Kernel to compute prefix sum and write back values
__global__ void writeBackKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, int numMatrices, int* freq_global, int* prefix_global, int max_freq_size, int max_padded_size) {
    int k = blockIdx.x;
    if (k >= numMatrices) return;

    int maxV = d_range[k];
    int rows = d_rows[k];
    int cols = d_cols[k];
    int totalElements = rows * cols;
    int n = maxV + 1;
    int p = 1;
    while (p < n) p *= 2;
    int padded_n = p;
    int* freqArray = &freq_global[k * max_freq_size];
    int* prefixSumArray = &prefix_global[k * max_padded_size];
    int tid = threadIdx.x;

    // Copy freqArray to prefixSumArray and pad
    for (int i = tid; i < padded_n; i += blockDim.x) {
        if (i < n) {
            prefixSumArray[i] = freqArray[i];
        } else {
            prefixSumArray[i] = 0;
        }
    }
    __syncthreads();

    // Blelloch scan
    // Up-sweep
    for (int stride = 1; stride < padded_n; stride *= 2) {
        for (int i = tid; i < padded_n / (2 * stride); i += blockDim.x) {
            int idx = (i + 1) * 2 * stride - 1;
            if (idx < padded_n) {
                prefixSumArray[idx] += prefixSumArray[idx - stride];
            }
        }
        __syncthreads();
    }

    // Down-sweep
    if (tid == 0) {
        prefixSumArray[padded_n - 1] = 0;
    }
    __syncthreads();

    for (int stride = padded_n / 2; stride > 0; stride /= 2) {
        for (int i = tid; i < padded_n / (2 * stride); i += blockDim.x) {
            int idx = (i + 1) * 2 * stride - 1;
            if (idx < padded_n) {
                int temp = prefixSumArray[idx - stride];
                prefixSumArray[idx - stride] = prefixSumArray[idx];
                prefixSumArray[idx] += temp;
            }
        }
        __syncthreads();
    }

    // Write back
    int offset = 0;
    for (int m = 0; m < k; m++) {
        offset += d_rows[m] * d_cols[m];
    }
    int* matrix = d_input + offset;

    for (int val = tid; val <= maxV; val += blockDim.x) {
        int start = prefixSumArray[val];
        int end = (val == maxV) ? totalElements : prefixSumArray[val + 1];
        for (int pos = start; pos < end; pos++) {
            matrix[pos] = val;
        }
    }
}


// RAII guard for CUDA pointers
struct CudaPtrGuard {
    void** ptr;
    explicit CudaPtrGuard(void** p) : ptr(p) {}
    ~CudaPtrGuard() { 
        if (*ptr) {
            cudaFree(*ptr); 
            *ptr = nullptr;
        }
    }
};

// Error checking wrapper
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error (" << msg << "): " 
             << cudaGetErrorString(err) << endl;
        throw runtime_error(cudaGetErrorString(err));
    }
}

vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, 
                                  vector<int>& range) {
    // Host memory management
    int* host_input = nullptr;
    vector<int> rows, cols;
    
    // Device pointers (all initialized to nullptr)
    int *d_input = nullptr, *d_range = nullptr, *d_rows = nullptr;
    int *d_cols = nullptr, *freq_global = nullptr;
    int *prefix_global = nullptr, *d_prefix_blocks = nullptr;

    try {
        // ====================== Host Setup ======================
        const int numMatrices = matrices.size();
        rows.resize(numMatrices);
        cols.resize(numMatrices);
        int totalElements = 0;

        // Validate input matrices
        for (int i = 0; i < numMatrices; i++) {
            if (matrices[i].empty() || matrices[i][0].empty()) {
                throw runtime_error("Empty matrix detected");
            }
            rows[i] = matrices[i].size();
            cols[i] = matrices[i][0].size();
            totalElements += rows[i] * cols[i];
        }

        // Flatten input matrices
        host_input = new int[totalElements];
        int pos = 0;
        for (int k = 0; k < numMatrices; k++) {
            for (const auto& row : matrices[k]) {
                for (int val : row) {
                    host_input[pos++] = val;
                }
            }
        }

        // ====================== Device Allocation ======================
        // Each allocation is immediately guarded
        checkCuda(cudaMalloc(&d_input, totalElements * sizeof(int)), "d_input alloc");
        CudaPtrGuard guard_d_input(reinterpret_cast<void**>(&d_input));

        checkCuda(cudaMalloc(&d_range, numMatrices * sizeof(int)), "d_range alloc");
        CudaPtrGuard guard_d_range(reinterpret_cast<void**>(&d_range));

        checkCuda(cudaMalloc(&d_rows, numMatrices * sizeof(int)), "d_rows alloc");
        CudaPtrGuard guard_d_rows(reinterpret_cast<void**>(&d_rows));

        checkCuda(cudaMalloc(&d_cols, numMatrices * sizeof(int)), "d_cols alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_cols));

        const int max_range = *max_element(range.begin(), range.end());
        const int max_freq_size = max_range + 1;
        const int max_padded_size = nextPowerOfTwo(max_freq_size);

        checkCuda(cudaMalloc(&freq_global, numMatrices * max_freq_size * sizeof(int)), 
                "freq_global alloc");
        CudaPtrGuard guard_freq_global(reinterpret_cast<void**>(&freq_global));

        checkCuda(cudaMalloc(&prefix_global, numMatrices * max_padded_size * sizeof(int)), 
                "prefix_global alloc");
        CudaPtrGuard guard_prefix_global(reinterpret_cast<void**>(&prefix_global));

        // ====================== Data Transfers ======================
        checkCuda(cudaMemcpy(d_input, host_input, totalElements * sizeof(int), 
                           cudaMemcpyHostToDevice), "d_input copy");
        checkCuda(cudaMemcpy(d_range, range.data(), numMatrices * sizeof(int),
                           cudaMemcpyHostToDevice), "d_range copy");
        checkCuda(cudaMemcpy(d_rows, rows.data(), numMatrices * sizeof(int),
                           cudaMemcpyHostToDevice), "d_rows copy");
        checkCuda(cudaMemcpy(d_cols, cols.data(), numMatrices * sizeof(int),
                           cudaMemcpyHostToDevice), "d_cols copy");

        // ====================== Kernel Setup ======================
        vector<int> prefix_blocks(numMatrices + 1, 0);
        for (int i = 0; i < numMatrices; ++i) {
            const int elements = rows[i] * cols[i];
            prefix_blocks[i+1] = prefix_blocks[i] + (elements + 1023)/1024;
        }

        checkCuda(cudaMalloc(&d_prefix_blocks, (numMatrices + 1) * sizeof(int)), 
                "d_prefix_blocks alloc");
        CudaPtrGuard guard_prefix_blocks(reinterpret_cast<void**>(&d_prefix_blocks));
        checkCuda(cudaMemcpy(d_prefix_blocks, prefix_blocks.data(), 
                           (numMatrices + 1) * sizeof(int), cudaMemcpyHostToDevice),
                           "d_prefix_blocks copy");

        // ====================== Kernel Execution ======================
        // Initialize frequency arrays
        initFreqKernel<<<numMatrices, 1024>>>(freq_global, d_range, max_freq_size, numMatrices);
        checkCuda(cudaGetLastError(), "initFreqKernel launch");
        checkCuda(cudaDeviceSynchronize(), "initFreqKernel sync");

        // Count frequencies
        countFreqKernel<<<prefix_blocks[numMatrices], 1024>>>(d_input, d_range, d_rows, d_cols,
                                                             d_prefix_blocks, numMatrices,
                                                             freq_global, max_freq_size, 1024);
        checkCuda(cudaGetLastError(), "countFreqKernel launch");
        checkCuda(cudaDeviceSynchronize(), "countFreqKernel sync");

        // Compute prefix sums and write back
        writeBackKernel<<<numMatrices, 1024>>>(d_input, d_range, d_rows, d_cols, numMatrices,
                                             freq_global, prefix_global, max_freq_size, 
                                             max_padded_size);
        checkCuda(cudaGetLastError(), "writeBackKernel launch");
        checkCuda(cudaDeviceSynchronize(), "writeBackKernel sync");

        // ====================== Retrieve Results ======================
        checkCuda(cudaMemcpy(host_input, d_input, totalElements * sizeof(int),
                           cudaMemcpyDeviceToHost), "results copy");

        // Update matrices
        pos = 0;
        for (int k = 0; k < numMatrices; k++) {
            const int r = rows[k], c = cols[k];
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    matrices[k][i][j] = host_input[pos++];
                }
            }
        }

        // ====================== Explicit Cleanup (Optional) ======================
        // Guards will auto-free, but explicit cleanup ensures early release
        cudaFree(d_input); d_input = nullptr;
        cudaFree(d_range); d_range = nullptr;
        cudaFree(d_rows); d_rows = nullptr;
        cudaFree(d_cols); d_cols = nullptr;
        cudaFree(freq_global); freq_global = nullptr;
        cudaFree(prefix_global); prefix_global = nullptr;
        cudaFree(d_prefix_blocks); d_prefix_blocks = nullptr;

        delete[] host_input;
        host_input = nullptr;

        return matrices;

    } catch (...) {
        // Cleanup host memory if error occurred before normal deletion
        if (host_input) {
            delete[] host_input;
            host_input = nullptr;
        }

		// Explicit device cleanup (complementary to RAII)
		if (d_input) cudaFree(d_input);
		if (d_range) cudaFree(d_range);
		if (d_rows) cudaFree(d_rows);
		if (d_cols) cudaFree(d_cols);
		if (freq_global) cudaFree(freq_global);
		if (prefix_global) cudaFree(prefix_global);
		if (d_prefix_blocks) cudaFree(d_prefix_blocks);

		// Nuclear option: Reset GPU to clean up any leaked context
		cudaError_t reset_err = cudaDeviceReset();
		if (reset_err != cudaSuccess) {
			cerr << "cudaDeviceReset() failed: " 
				 << cudaGetErrorString(reset_err) << endl;
		}
        
        // Re-throw to notify caller
        throw;
    }
}