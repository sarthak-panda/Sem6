#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
// #include "modify.cuh"

using namespace std;

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

__device__ int deviceNextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

__global__ void countFreqKernel(int* d_input, const int* d_range, const int* d_offsets, 
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
    int elements = d_offsets[matrix_k+1] - d_offsets[matrix_k];
    int* freqArray = &prefix_global[matrix_k * max_padded_size];
    int* matrix = d_input + d_offsets[matrix_k];

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

__global__ void computePrefixSumsKernel(int* prefix_global, const int* d_range, const int* d_offsets,
                                       int numMatrices, int max_padded_size) {
    int k = blockIdx.x;
    if (k >= numMatrices) return;
    
    int maxV = d_range[k];
    int n = maxV + 1;
    int padded_n = deviceNextPowerOfTwo(n);
    if (padded_n > max_padded_size) return;

    int* prefixSumArray = &prefix_global[k * max_padded_size];
    int tid = threadIdx.x;

    // Initialize padded elements to 0
    for (int i = tid; i < padded_n; i += blockDim.x) {
        prefixSumArray[i] = (i < n) ? prefixSumArray[i] : 0;
    }
    __syncthreads();

    // Blelloch scan upsweep
    for (int stride = 1; stride < padded_n; stride *= 2) {
        for (int i = tid; i < padded_n / (2 * stride); i += blockDim.x) {
            int idx = (i + 1) * 2 * stride - 1;
            if (idx < padded_n) {
                prefixSumArray[idx] += prefixSumArray[idx - stride];
            }
        }
        __syncthreads();
    }

    // Downsweep preparation
    if (tid == 0) {
        prefixSumArray[padded_n - 1] = 0;
    }
    __syncthreads();

    // Blelloch scan downsweep
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
}

__global__ void writeBackKernel(int* d_input, const int* d_range, const int* d_offsets,
                                int numMatrices, int* prefix_global, int max_padded_size) {
    int k = blockIdx.y;
    if (k >= numMatrices) return;
    
    int maxV = d_range[k];
    int totalElements = d_offsets[k+1] - d_offsets[k];
    int* prefixSumArray = &prefix_global[k * max_padded_size];
    int* matrix = d_input + d_offsets[k];

    // Process values in parallel across blocks
    int val = blockIdx.x * blockDim.x + threadIdx.x;
    if (val > maxV) return;

    int start = prefixSumArray[val];
    int end = (val == maxV) ? totalElements : prefixSumArray[val + 1];
    
    // Parallelize position writes within value range
    for (int pos = start + threadIdx.y; pos < end; pos += blockDim.y) {
        if (pos < end) matrix[pos] = val;
    }
}

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << endl;
        throw runtime_error(cudaGetErrorString(err));
    }
}

vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& range) {
    int* host_input = nullptr;
    int *d_input = nullptr, *d_range = nullptr, *d_offsets = nullptr;
    int *prefix_global = nullptr, *d_prefix_blocks = nullptr;
    vector<int> offsets;

    try {
        const int numMatrices = matrices.size();
        vector<int> rows(numMatrices), cols(numMatrices);
        int totalElements = 0;

        // Calculate offsets and copy data to flat host array
        offsets.push_back(0);
        for (int i = 0; i < numMatrices; i++) {
            rows[i] = matrices[i].size();
            cols[i] = matrices[i][0].size();
            int elements = rows[i] * cols[i];
            totalElements += elements;
            offsets.push_back(totalElements);
        }

        host_input = new int[totalElements];
        int pos = 0;
        for (int k = 0; k < numMatrices; k++) {
            for (const auto& row : matrices[k]) {
                for (int val : row) {
                    host_input[pos++] = val;
                }
            }
        }

        // Allocate and copy device memory
        checkCuda(cudaMalloc(&d_input, totalElements * sizeof(int)), "d_input alloc");
        CudaPtrGuard guard_d_input(reinterpret_cast<void**>(&d_input));
        checkCuda(cudaMemcpy(d_input, host_input, totalElements * sizeof(int), cudaMemcpyHostToDevice), "Input copy");

        checkCuda(cudaMalloc(&d_range, numMatrices * sizeof(int)), "d_range alloc");
        CudaPtrGuard guard_d_range(reinterpret_cast<void**>(&d_range));
        checkCuda(cudaMemcpy(d_range, range.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "Range copy");

        checkCuda(cudaMalloc(&d_offsets, (numMatrices+1) * sizeof(int)), "d_offsets alloc");
        CudaPtrGuard guard_d_offsets(reinterpret_cast<void**>(&d_offsets));
        checkCuda(cudaMemcpy(d_offsets, offsets.data(), (numMatrices+1) * sizeof(int), cudaMemcpyHostToDevice), "Offsets copy");

        // Calculate max padded size
        int max_range = *max_element(range.begin(), range.end());
        int max_padded_size = 1;
        while (max_padded_size <= max_range) max_padded_size *= 2;

        checkCuda(cudaMalloc(&prefix_global, numMatrices * max_padded_size * sizeof(int)), "prefix_global alloc");
        CudaPtrGuard guard_prefix_global(reinterpret_cast<void**>(&prefix_global));
        checkCuda(cudaMemset(prefix_global, 0, numMatrices * max_padded_size * sizeof(int)), "prefix_global init");

        // Calculate prefix blocks
        vector<int> prefix_blocks(numMatrices + 1, 0);
        for (int i = 0; i < numMatrices; ++i) {
            int elements = offsets[i+1] - offsets[i];
            prefix_blocks[i+1] = prefix_blocks[i] + (elements + 1023) / 1024;
        }

        checkCuda(cudaMalloc(&d_prefix_blocks, (numMatrices + 1) * sizeof(int)), "d_prefix_blocks alloc");
        CudaPtrGuard guard_prefix_blocks(reinterpret_cast<void**>(&d_prefix_blocks));
        checkCuda(cudaMemcpy(d_prefix_blocks, prefix_blocks.data(), (numMatrices + 1) * sizeof(int), cudaMemcpyHostToDevice), "prefix_blocks copy");

        // Launch kernels
        countFreqKernel<<<prefix_blocks[numMatrices], 1024>>>(d_input, d_range, d_offsets, d_prefix_blocks, 
                                                             numMatrices, prefix_global, max_padded_size, 1024);
        checkCuda(cudaGetLastError(), "countFreqKernel");
        checkCuda(cudaDeviceSynchronize(), "countFreq sync");

        computePrefixSumsKernel<<<numMatrices, 1024>>>(prefix_global, d_range, d_offsets, numMatrices, max_padded_size);
        checkCuda(cudaGetLastError(), "prefixSumKernel");
        checkCuda(cudaDeviceSynchronize(), "prefixSum sync");

        // Configure writeback kernel with 2D grid (values x matrices)
        dim3 blockSize(256, 1);
        dim3 gridSize((max_range + blockSize.x) / blockSize.x, numMatrices);
        writeBackKernel<<<gridSize, blockSize>>>(d_input, d_range, d_offsets, numMatrices, prefix_global, max_padded_size);
        checkCuda(cudaGetLastError(), "writeBackKernel");
        checkCuda(cudaDeviceSynchronize(), "writeBack sync");

        // Copy results back
        checkCuda(cudaMemcpy(host_input, d_input, totalElements * sizeof(int), cudaMemcpyDeviceToHost), "results copy");
        pos = 0;
        for (int k = 0; k < numMatrices; k++) {
            const int r = rows[k], c = cols[k];
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    matrices[k][i][j] = host_input[pos++];
                }
            }
        }

        delete[] host_input;
        return matrices;
    } catch (...) {
        if (host_input) delete[] host_input;
        cudaDeviceReset();
        throw;
    }
}