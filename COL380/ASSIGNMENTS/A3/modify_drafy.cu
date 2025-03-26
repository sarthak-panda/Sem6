#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

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

    //for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int i = start + threadIdx.x;
        if(i<end){
            int val = matrix[i];
            if (val <= maxV) {
                atomicAdd(&freqArray[val], 1);
            }
        }
    //}
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

vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& range) {
    int numMatrices = matrices.size();
    vector<int> rows(numMatrices), cols(numMatrices);
    int totalElements = 0;
    for (int i = 0; i < numMatrices; i++) {
        rows[i] = matrices[i].size();
        cols[i] = matrices[i][0].size();
        totalElements += rows[i] * cols[i];
    }

    int* input = new int[totalElements];
    int pos = 0;
    for (int k = 0; k < numMatrices; k++) {
        for (auto& row : matrices[k]) {
            for (int val : row) {
                input[pos++] = val;
            }
        }
    }

    // Compute max_freq_size and max_padded_size
    int max_range = *max_element(range.begin(), range.end());
    int max_freq_size = max_range + 1;
    int max_padded_size = nextPowerOfTwo(max_freq_size);

    // Allocate device memory
    int *d_input, *d_range, *d_rows, *d_cols, *freq_global, *prefix_global, *d_prefix_blocks;
    cudaMalloc(&d_input, totalElements * sizeof(int));
    cudaMalloc(&d_range, numMatrices * sizeof(int));
    cudaMalloc(&d_rows, numMatrices * sizeof(int));
    cudaMalloc(&d_cols, numMatrices * sizeof(int));
    cudaMalloc(&freq_global, numMatrices * max_freq_size * sizeof(int));
    cudaMalloc(&prefix_global, numMatrices * max_padded_size * sizeof(int));

    cudaMemcpy(d_input, input, totalElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_range, range.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, rows.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);

    // Compute blocks per matrix and prefix_blocks
    vector<int> blocks_per_matrix(numMatrices);
    vector<int> prefix_blocks(numMatrices + 1, 0);
    for (int i = 0; i < numMatrices; ++i) {
        int elements = rows[i] * cols[i];
        blocks_per_matrix[i] = (elements + 1023) / 1024; // Using 1024 threads per block
        prefix_blocks[i+1] = prefix_blocks[i] + blocks_per_matrix[i];
    }
    cudaMalloc(&d_prefix_blocks, (numMatrices + 1) * sizeof(int));
    cudaMemcpy(d_prefix_blocks, prefix_blocks.data(), (numMatrices + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernels
    int threadsPerBlock = 1024;
    initFreqKernel<<<numMatrices, threadsPerBlock>>>(freq_global, d_range, max_freq_size, numMatrices);
    countFreqKernel<<<prefix_blocks[numMatrices], threadsPerBlock>>>(d_input, d_range, d_rows, d_cols, d_prefix_blocks, numMatrices, freq_global, max_freq_size, threadsPerBlock);
    writeBackKernel<<<numMatrices, threadsPerBlock>>>(d_input, d_range, d_rows, d_cols, numMatrices, freq_global, prefix_global, max_freq_size, max_padded_size);

    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(input, d_input, totalElements * sizeof(int), cudaMemcpyDeviceToHost);

    // Update matrices
    pos = 0;
    for (int k = 0; k < numMatrices; k++) {
        int r = rows[k], c = cols[k];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                matrices[k][i][j] = input[pos++];
            }
        }
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_range);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(freq_global);
    cudaFree(prefix_global);
    cudaFree(d_prefix_blocks);
    delete[] input;

    return matrices;
}
