#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
using namespace std;

// Helper function to compute prefix sums
vector<int> computePrefixSum(const vector<int>& sizes) {
    vector<int> prefix(sizes.size() + 1, 0);
    for (int i = 0; i < sizes.size(); ++i)
        prefix[i+1] = prefix[i] + sizes[i];
    return prefix;
}

// Kernel to initialize frequency arrays
__global__ void initFreqKernel(int* prefix_global, const int* d_range, const int* d_prefix_freq, const int* d_prefix_blocks_init int numMatrices) {
    int block_idx = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_init[i] <= block_idx && block_idx < d_prefix_blocks_init[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;

    int maxV = d_range[matrix_k];
    int* freqArray = &prefix_global[d_prefix_freq[matrix_k]];
    int elements = maxV + 1;
    
    int block_offset = block_idx - d_prefix_blocks_init[matrix_k];
    int tid = block_offset * blockDim.x + threadIdx.x;
    if (tid < elements) freqArray[tid] = 0;
}

// Kernel to count frequencies
__global__ void countFreqKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, const int* d_prefix_blocks_count const int* d_prefix_freq, int numMatrices, int* prefix_global) {
    int block_idx = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_count[i] <= block_idx && block_idx < d_prefix_blocks_count[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;

    int maxV = d_range[matrix_k];
    int rows = d_rows[matrix_k];
    int cols = d_cols[matrix_k];
    int elements = rows * cols;
    int* freqArray = &prefix_global[d_prefix_freq[matrix_k]];
    
    int block_offset = block_idx - d_prefix_blocks_count[matrix_k];
    int start = block_offset * blockDim.x;
    int end = min(start + blockDim.x, elements);
    
    int input_offset = 0;
    for (int m = 0; m < matrix_k; ++m)
        input_offset += d_rows[m] * d_cols[m];
    
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int val = d_input[input_offset + i];
        if (val <= maxV) atomicAdd(&freqArray[val], 1);
    }
}

// Distributed Blelloch scan implementation
__global__ void localScanKernel(int* prefix_global, const int* d_range const int* d_prefix_freq, const int* d_prefix_blocks_scan int* chunk_sums, int numMatrices) {
    int block_idx = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_scan[i] <= block_idx && block_idx < d_prefix_blocks_scan[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;

    int maxV = d_range[matrix_k];
    int n = maxV + 1;
    int* arr = &prefix_global[d_prefix_freq[matrix_k]];
    
    int chunk_idx = block_idx - d_prefix_blocks_scan[matrix_k];
    int chunk_size = 1024;
    int start = chunk_idx * chunk_size;
    int end = min(start + chunk_size, n);
    int tid = threadIdx.x;

    // Local upsweep
    for (int stride = 1; stride < chunk_size; stride *= 2) {
        int idx = start + tid * stride * 2 + stride - 1;
        if (idx < end && (idx - stride) >= start)
            arr[idx] += arr[idx - stride];
        __syncthreads();
    }

    // Store chunk sum
    if (tid == 0 && end > start) {
        chunk_sums[block_idx] = arr[end-1];
        arr[end-1] = 0;
    }
    __syncthreads();

    // Local downsweep
    for (int stride = chunk_size/2; stride >= 1; stride /= 2) {
        int idx = start + tid * stride * 2 + stride - 1;
        if (idx < end) {
            int temp = arr[idx - stride];
            arr[idx - stride] = arr[idx];
            arr[idx] += temp;
        }
        __syncthreads();
    }
}

__global__ void globalOffsetKernel(int* chunk_sums, const int* d_prefix_blocks_scan, int numMatrices) {
    int matrix_k = blockIdx.x;
    if (matrix_k >= numMatrices) return;
    
    int start = d_prefix_blocks_scan[matrix_k];
    int end = d_prefix_blocks_scan[matrix_k+1];
    int n = end - start;
    
    extern __shared__ int shared_sums[];
    int tid = threadIdx.x;
    
    if (tid < n) shared_sums[tid] = chunk_sums[start + tid];
    else shared_sums[tid] = 0;
    __syncthreads();

    // Blelloch scan in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) shared_sums[tid] += shared_sums[tid - stride];
        __syncthreads();
    }

    if (tid < n) chunk_sums[start + tid] = shared_sums[tid];
}

__global__ void applyOffsetKernel(int* prefix_global, const int* d_range, const int* d_prefix_freq, const int* d_prefix_blocks_scan, const int* chunk_sums, int numMatrices) {
    int block_idx = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_scan[i] <= block_idx && block_idx < d_prefix_blocks_scan[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;

    int maxV = d_range[matrix_k];
    int* arr = &prefix_global[d_prefix_freq[matrix_k]];
    int chunk_offset = chunk_sums[block_idx];
    
    int chunk_idx = block_idx - d_prefix_blocks_scan[matrix_k];
    int chunk_size = 1024;
    int start = chunk_idx * chunk_size;
    int end = min(start + chunk_size, maxV + 1);
    
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        arr[i] += chunk_offset;
    }
}

// Distributed write-back kernel
__global__ void writeBackKernel(int* d_input, const int* d_range, const int* d_rows const int* d_cols, const int* d_prefix_blocks_write const int* d_prefix_freq, int numMatrices int* prefix_global) {
    int block_idx = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_write[i] <= block_idx && block_idx < d_prefix_blocks_write[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;

    int maxV = d_range[matrix_k];
    int* prefixArr = &prefix_global[d_prefix_freq[matrix_k]];
    int elements = d_rows[matrix_k] * d_cols[matrix_k];
    
    int block_offset = block_idx - d_prefix_blocks_write[matrix_k];
    int val_start = block_offset * blockDim.x;
    int val_end = min(val_start + blockDim.x, maxV + 1);
    
    int input_offset = 0;
    for (int m = 0; m < matrix_k; ++m)
        input_offset += d_rows[m] * d_cols[m];
    
    for (int val = val_start + threadIdx.x; val < val_end; val += blockDim.x) {
        int start = prefixArr[val];
        int end = (val == maxV) ? elements : prefixArr[val + 1];
        
        int* matrix = d_input + input_offset;
        for (int pos = start; pos < end; ++pos)
            matrix[pos] = val;
    }
}

struct CudaPtrGuard {
    void** ptr;
    explicit CudaPtrGuard(void** p) : ptr(p) {}
    ~CudaPtrGuard() { 
        if (ptr && *ptr) cudaFree(*ptr); 
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
    int *d_input = nullptr, *d_range = nullptr, *d_rows = nullptr, *d_cols = nullptr;
    int *prefix_global = nullptr, *d_prefix_freq = nullptr;
    int *d_prefix_blocks_init = nullptr, *d_prefix_blocks_count = nullptr;
    int *d_prefix_blocks_scan = nullptr, *d_prefix_blocks_write = nullptr;
    int *chunk_sums = nullptr;

    try {
        const int numMatrices = matrices.size();
        vector<int> rows(numMatrices), cols(numMatrices), freq_sizes(numMatrices);
        int totalElements = 0;

        // Initialize metadata
        for (int i = 0; i < numMatrices; ++i) {
            rows[i] = matrices[i].size();
            cols[i] = matrices[i][0].size();
            totalElements += rows[i] * cols[i];
            freq_sizes[i] = range[i] + 1;
        }

        // Prepare host input
        host_input = new int[totalElements];
        int pos = 0;
        for (const auto& m : matrices)
            for (const auto& r : m)
                for (int v : r)
                    host_input[pos++] = v;

        // Compute prefix sums for frequency arrays
        vector<int> prefix_freq = computePrefixSum(freq_sizes);
        const int total_freq_size = prefix_freq.back();

        // Allocate device memory
        checkCuda(cudaMalloc(&d_input, totalElements * sizeof(int)), "d_input");
        checkCuda(cudaMalloc(&d_range, numMatrices * sizeof(int)), "d_range");
        checkCuda(cudaMalloc(&d_rows, numMatrices * sizeof(int)), "d_rows");
        checkCuda(cudaMalloc(&d_cols, numMatrices * sizeof(int)), "d_cols");
        checkCuda(cudaMalloc(&prefix_global, total_freq_size * sizeof(int)), "prefix_global");
        checkCuda(cudaMalloc(&d_prefix_freq, (numMatrices+1)*sizeof(int)), "d_prefix_freq");

        // Copy data to device
        checkCuda(cudaMemcpy(d_input, host_input, totalElements*sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_range, range.data(), numMatrices*sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rows, rows.data(), numMatrices*sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_cols, cols.data(), numMatrices*sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_prefix_freq, prefix_freq.data(), (numMatrices+1)*sizeof(int), cudaMemcpyHostToDevice));

        // Initialize frequency arrays
        vector<int> prefix_blocks_init(numMatrices + 1, 0);
        for (int i = 0; i < numMatrices; ++i) {
            int blocks = (freq_sizes[i] + 1023) / 1024;
            prefix_blocks_init[i+1] = prefix_blocks_init[i] + blocks;
        }
        checkCuda(cudaMalloc(&d_prefix_blocks_init, (numMatrices+1)*sizeof(int)), "d_prefix_blocks_init");
        checkCuda(cudaMemcpy(d_prefix_blocks_init, prefix_blocks_init.data(), (numMatrices+1)*sizeof(int), cudaMemcpyHostToDevice));
        initFreqKernel<<<prefix_blocks_init.back(), 1024>>>(prefix_global, d_range, d_prefix_freq, d_prefix_blocks_init, numMatrices);
        checkCuda(cudaDeviceSynchronize(), "initFreqKernel");

        // Count frequencies
        vector<int> prefix_blocks_count(numMatrices + 1, 0);
        for (int i = 0; i < numMatrices; ++i) {
            int blocks = (rows[i] * cols[i] + 1023) / 1024;
            prefix_blocks_count[i+1] = prefix_blocks_count[i] + blocks;
        }
        checkCuda(cudaMalloc(&d_prefix_blocks_count, (numMatrices+1)*sizeof(int)), "d_prefix_blocks_count");
        checkCuda(cudaMemcpy(d_prefix_blocks_count, prefix_blocks_count.data(), (numMatrices+1)*sizeof(int), cudaMemcpyHostToDevice));
        countFreqKernel<<<prefix_blocks_count.back(), 1024>>>(d_input, d_range, d_rows, d_cols, d_prefix_blocks_count, d_prefix_freq, numMatrices, prefix_global);
        checkCuda(cudaDeviceSynchronize(), "countFreqKernel");

        // Distributed prefix sum computation
        vector<int> prefix_blocks_scan(numMatrices + 1, 0);
        for (int i = 0; i < numMatrices; ++i) {
            int chunks = (freq_sizes[i] + 1023) / 1024;
            prefix_blocks_scan[i+1] = prefix_blocks_scan[i] + chunks;
        }
        checkCuda(cudaMalloc(&d_prefix_blocks_scan, (numMatrices+1)*sizeof(int)), "d_prefix_blocks_scan");
        checkCuda(cudaMemcpy(d_prefix_blocks_scan, prefix_blocks_scan.data(), (numMatrices+1)*sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc(&chunk_sums, prefix_blocks_scan.back() * sizeof(int)), "chunk_sums");

        // Phase 1: Local chunk scans
        localScanKernel<<<prefix_blocks_scan.back(), 1024>>>(prefix_global, d_range, d_prefix_freq, d_prefix_blocks_scan, chunk_sums, numMatrices);
        checkCuda(cudaDeviceSynchronize(), "localScanKernel");

        // Phase 2: Compute global offsets
        globalOffsetKernel<<<numMatrices, 1024, 1024*sizeof(int)>>>(chunk_sums, d_prefix_blocks_scan, numMatrices);
        checkCuda(cudaDeviceSynchronize(), "globalOffsetKernel");

        // Phase 3: Apply offsets
        applyOffsetKernel<<<prefix_blocks_scan.back(), 1024>>>(prefix_global, d_range, d_prefix_freq, d_prefix_blocks_scan, chunk_sums, numMatrices);
        checkCuda(cudaDeviceSynchronize(), "applyOffsetKernel");

        // Write back sorted values
        vector<int> prefix_blocks_write(numMatrices + 1, 0);
        for (int i = 0; i < numMatrices; ++i) {
            int blocks = (range[i] + 1 + 1023) / 1024;
            prefix_blocks_write[i+1] = prefix_blocks_write[i] + blocks;
        }
        checkCuda(cudaMalloc(&d_prefix_blocks_write, (numMatrices+1)*sizeof(int)), "d_prefix_blocks_write");
        checkCuda(cudaMemcpy(d_prefix_blocks_write, prefix_blocks_write.data(), (numMatrices+1)*sizeof(int), cudaMemcpyHostToDevice));
        writeBackKernel<<<prefix_blocks_write.back(), 1024>>>(d_input, d_range, d_rows, d_cols, d_prefix_blocks_write, d_prefix_freq, numMatrices, prefix_global);
        checkCuda(cudaDeviceSynchronize(), "writeBackKernel");

        // Copy results back
        checkCuda(cudaMemcpy(host_input, d_input, totalElements*sizeof(int), cudaMemcpyDeviceToHost));

        // Update matrices
        pos = 0;
        for (int k = 0; k < numMatrices; ++k) {
            auto& m = matrices[k];
            for (auto& r : m)
                for (int& v : r)
                    v = host_input[pos++];
        }

        // Cleanup
        delete[] host_input;
        cudaFree(prefix_global);
        cudaFree(d_prefix_freq);
        cudaFree(d_prefix_blocks_init);
        cudaFree(d_prefix_blocks_count);
        cudaFree(d_prefix_blocks_scan);
        cudaFree(d_prefix_blocks_write);
        cudaFree(chunk_sums);

        return matrices;
    } catch (...) {
        if (host_input) delete[] host_input;
        cudaFree(d_input);
        cudaFree(d_range);
        cudaFree(d_rows);
        cudaFree(d_cols);
        cudaFree(prefix_global);
        cudaFree(d_prefix_freq);
        cudaFree(d_prefix_blocks_init);
        cudaFree(d_prefix_blocks_count);
        cudaFree(d_prefix_blocks_scan);
        cudaFree(d_prefix_blocks_write);
        cudaFree(chunk_sums);
        throw;
    }
}