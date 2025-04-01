#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "modify.cuh"
using namespace std;
int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}
__global__ void countFreqKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, 
                                const int* d_prefix_blocks, int numMatrices, int* prefix_global, 
                                int max_padded_size, int threadsPerBlock) {
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
    int* freqArray = &prefix_global[matrix_k * max_padded_size];
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
__global__ void writeBackKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, 
                                int numMatrices, int* prefix_global, int max_padded_size) {
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
    int* prefixSumArray = &prefix_global[k * max_padded_size];
    int tid = threadIdx.x;
    for (int i = tid; i < padded_n; i += blockDim.x) {
        if (i >= n) {
            prefixSumArray[i] = 0;
        }
    }
    __syncthreads();
    for (int stride = 1; stride < padded_n; stride *= 2) {
        for (int i = tid; i < padded_n / (2 * stride); i += blockDim.x) {
            int idx = (i + 1) * 2 * stride - 1;
            if (idx < padded_n) {
                prefixSumArray[idx] += prefixSumArray[idx - stride];
            }
        }
        __syncthreads();
    }
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
    int offset = 0;
    for (int m = 0; m < k; m++) {
        offset += d_rows[m] * d_cols[m];
    }
    int* matrix = d_input + offset;
    for (int val = tid; val <= maxV; val += blockDim.x) {
        int start = prefixSumArray[val];
        int end = (val == maxV) ? totalElements : prefixSumArray[val + 1];
        if (start >= totalElements || end > totalElements) continue;
        for (int pos = start; pos < end; pos++) {
            matrix[pos] = val;
        }
    }
}
struct CudaPtrGuard {
    void** ptr;
    explicit CudaPtrGuard(void** p) : ptr(p) {}
    ~CudaPtrGuard() { 
        if (ptr && *ptr) {
            cudaFree(*ptr); 
            *ptr = nullptr;
        }
    }
};
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << endl;
        throw runtime_error(cudaGetErrorString(err));
    }
}
vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& range) {
    int* host_input = nullptr;
    int *h_range = nullptr, *h_rows = nullptr, *h_cols = nullptr, *h_prefix_blocks = nullptr;
    cudaStream_t stream = nullptr;
    int *d_input = nullptr, *d_range = nullptr, *d_rows = nullptr;
    int *d_cols = nullptr, *prefix_global = nullptr, *d_prefix_blocks = nullptr;

    try {
        const int numMatrices = matrices.size();
        vector<int> rows(numMatrices), cols(numMatrices);
        int totalElements = 0;

        for (int i = 0; i < numMatrices; i++) {
            if (matrices[i].empty() || matrices[i][0].empty()) {
                throw runtime_error("Empty matrix detected");
            }
            rows[i] = matrices[i].size();
            cols[i] = matrices[i][0].size();
            totalElements += rows[i] * cols[i];
        }

        // Allocate pinned host memory
        checkCuda(cudaMallocHost(&host_input, totalElements * sizeof(int)), "host_input pinned alloc");
        checkCuda(cudaMallocHost(&h_range, numMatrices * sizeof(int)), "h_range pinned alloc");
        checkCuda(cudaMallocHost(&h_rows, numMatrices * sizeof(int)), "h_rows pinned alloc");
        checkCuda(cudaMallocHost(&h_cols, numMatrices * sizeof(int)), "h_cols pinned alloc");

        // Initialize host arrays
        memcpy(h_range, range.data(), numMatrices * sizeof(int));
        memcpy(h_rows, rows.data(), numMatrices * sizeof(int));
        memcpy(h_cols, cols.data(), numMatrices * sizeof(int));

        int pos = 0;
        for (int k = 0; k < numMatrices; k++) {
            for (const auto& row : matrices[k]) {
                for (int val : row) {
                    host_input[pos++] = val;
                }
            }
        }
		// int pos = 0;
		// for (int k = 0; k < numMatrices; k++) {
		// 	const int numElements = rows[k] * cols[k];
		// 	// Assuming matrices[k] points to a contiguous block, e.g., matrices[k][0]
		// 	std::copy(matrices[k][0], matrices[k][0] + numElements, host_input + pos);
		// 	pos += numElements;
		// }


        // Allocate device memory
        checkCuda(cudaMalloc(&d_input, totalElements * sizeof(int)), "d_input alloc");
        CudaPtrGuard guard_d_input(reinterpret_cast<void**>(&d_input));
        checkCuda(cudaMalloc(&d_range, numMatrices * sizeof(int)), "d_range alloc");
        CudaPtrGuard guard_d_range(reinterpret_cast<void**>(&d_range));
        checkCuda(cudaMalloc(&d_rows, numMatrices * sizeof(int)), "d_rows alloc");
        CudaPtrGuard guard_d_rows(reinterpret_cast<void**>(&d_rows));
        checkCuda(cudaMalloc(&d_cols, numMatrices * sizeof(int)), "d_cols alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_cols));

        // Compute max padded size
        const int max_range = *max_element(range.begin(), range.end());
        const int max_freq_size = max_range + 1;
        const int max_padded_size = nextPowerOfTwo(max_freq_size);
        checkCuda(cudaMalloc(&prefix_global, numMatrices * max_padded_size * sizeof(int)), "prefix_global alloc");
        CudaPtrGuard guard_prefix_global(reinterpret_cast<void**>(&prefix_global));

        // Compute prefix_blocks
        vector<int> prefix_blocks(numMatrices + 1, 0);
        for (int i = 0; i < numMatrices; ++i) {
            const int elements = rows[i] * cols[i];
            prefix_blocks[i+1] = prefix_blocks[i] + (elements + 1023)/1024;
        }

        checkCuda(cudaMallocHost(&h_prefix_blocks, (numMatrices + 1) * sizeof(int)), "h_prefix_blocks pinned alloc");
        memcpy(h_prefix_blocks, prefix_blocks.data(), (numMatrices + 1) * sizeof(int));

        checkCuda(cudaMalloc(&d_prefix_blocks, (numMatrices + 1) * sizeof(int)), "d_prefix_blocks alloc");
        CudaPtrGuard guard_prefix_blocks(reinterpret_cast<void**>(&d_prefix_blocks));

        // Create CUDA stream
        checkCuda(cudaStreamCreate(&stream), "stream creation");

        // Asynchronous memory copies (Host to Device)
        checkCuda(cudaMemcpyAsync(d_input, host_input, totalElements * sizeof(int), cudaMemcpyHostToDevice, stream), "d_input copy");
        checkCuda(cudaMemcpyAsync(d_range, h_range, numMatrices * sizeof(int), cudaMemcpyHostToDevice, stream), "d_range copy");
        checkCuda(cudaMemcpyAsync(d_rows, h_rows, numMatrices * sizeof(int), cudaMemcpyHostToDevice, stream), "d_rows copy");
        checkCuda(cudaMemcpyAsync(d_cols, h_cols, numMatrices * sizeof(int), cudaMemcpyHostToDevice, stream), "d_cols copy");
        checkCuda(cudaMemcpyAsync(d_prefix_blocks, h_prefix_blocks, (numMatrices + 1) * sizeof(int), cudaMemcpyHostToDevice, stream), "d_prefix_blocks copy");

        // Launch kernels in the same stream
        countFreqKernel<<<prefix_blocks[numMatrices], 1024, 0, stream>>>(d_input, d_range, d_rows, d_cols, d_prefix_blocks, numMatrices, prefix_global, max_padded_size, 1024);
        checkCuda(cudaGetLastError(), "countFreqKernel launch");

        writeBackKernel<<<numMatrices, 1024, 0, stream>>>(d_input, d_range, d_rows, d_cols, numMatrices, prefix_global, max_padded_size);
        checkCuda(cudaGetLastError(), "writeBackKernel launch");

        // Asynchronous memory copy (Device to Host)
        checkCuda(cudaMemcpyAsync(host_input, d_input, totalElements * sizeof(int), cudaMemcpyDeviceToHost, stream), "results copy");

        // Synchronize stream
        checkCuda(cudaStreamSynchronize(stream), "stream sync");

        // Copy results back to matrices
        pos = 0;
        for (int k = 0; k < numMatrices; k++) {
            const int r = rows[k], c = cols[k];
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    matrices[k][i][j] = host_input[pos++];
                }
            }
        }
		// for (int k = 0; k < numMatrices; k++) {
		// 	const int numElements = rows[k] * cols[k];
		// 	// Assuming matrices[k] points to a contiguous block (e.g., matrices[k][0])
		// 	std::copy(host_input + pos, host_input + pos + numElements, matrices[k][0]);
		// 	pos += numElements;
		// }
		

        // Cleanup pinned host memory
        if (host_input) {
            cudaFreeHost(host_input);
            host_input = nullptr;
        }
        if (h_range) {
            cudaFreeHost(h_range);
            h_range = nullptr;
        }
        if (h_rows) {
            cudaFreeHost(h_rows);
            h_rows = nullptr;
        }
        if (h_cols) {
            cudaFreeHost(h_cols);
            h_cols = nullptr;
        }
        if (h_prefix_blocks) {
            cudaFreeHost(h_prefix_blocks);
            h_prefix_blocks = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        return matrices;
    } catch (...) {
        // Cleanup pinned memory and stream
        if (host_input) cudaFreeHost(host_input);
        if (h_range) cudaFreeHost(h_range);
        if (h_rows) cudaFreeHost(h_rows);
        if (h_cols) cudaFreeHost(h_cols);
        if (h_prefix_blocks) cudaFreeHost(h_prefix_blocks);
        if (stream) cudaStreamDestroy(stream);
        // Device memory cleanup handled by CudaPtrGuard
        cudaDeviceReset();
        throw;
    }
}