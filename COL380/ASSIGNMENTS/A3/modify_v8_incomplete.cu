#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "modify.cuh"
#include <cmath>

using namespace std;
int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}
// __global__ void initFreqKernel(int* prefix_global, const int* d_range, int max_padded_size, int numMatrices) {
//     int k = blockIdx.x;
//     if (k >= numMatrices) return;
//     int maxV = d_range[k];
//     int* freqArray = &prefix_global[k * max_padded_size];
//     int tid = threadIdx.x;
//     for (int i = tid; i <= maxV; i += blockDim.x) {
//         freqArray[i] = 0;
//     }
// }
__global__ void countFreqKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, 
                                const int* d_prefix_blocks_pass_0,const int* d_prefix_indices_pass_0, int numMatrices, int* prefix_global, 
                                int max_padded_size, int threadsPerBlock) {
    int block_idx = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_pass_0[i] <= block_idx && block_idx < d_prefix_blocks_pass_0[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;
    int maxV = d_range[matrix_k];
    int elements = d_rows[matrix_k] * d_cols[matrix_k];
    int startIndexOfFreqArray = d_prefix_indices_pass_0[matrix_k];
    int* freqArray = &prefix_global[startIndexOfFreqArray];
    int offset = 0;
    for (int m = 0; m < matrix_k; m++) {
        offset += d_rows[m] * d_cols[m];
    }
    int* matrix = d_input + offset;
    int blocks_before = block_idx - d_prefix_blocks_pass_0[matrix_k];
    int start = blocks_before * threadsPerBlock;
    // int end = min(start + threadsPerBlock, elements);
    // for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
    int i = start + threadIdx.x;
    int val = matrix[i];
    if (val <= maxV) {
        atomicAdd(&freqArray[val], 1);
    }
    //     break;
    // }
}

__device__ __forceinline__ void inclusiveBlellochScan(int* array, int array_size){
    int tid=threadIdx.x;
    int myVal = array[tid];
    for (int stride = 1; stride < array_size; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < array_size) {
            array[idx] += array[idx - stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        array[array_size - 1] = 0;
    }
    __syncthreads();
    for (int stride = array_size / 2; stride > 0; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < array_size) {
            int temp = array[idx - stride];
            array[idx - stride] = array[idx];
            array[idx] += temp;
        }
        __syncthreads();
    }
    array[tid] += myVal;
    __syncthreads();
}

__global__ void preFixSumKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, 
                                int* prefix_global, const int* d_prefix_blocks_pass_num,const int* d_prefix_indices_pass_num,const int* d_prefix_indices_pass_num_next,int numThreadsPerBlock, int numMatrices, int pass_num) {
    int k = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_pass_num[i] <= block_idx && block_idx < d_prefix_blocks_pass_num[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;
    if(pass_num==0){
        //just calculate belloch prefix for the array corresponding to that block(inclusive belloch)
        //write the end value of each block at appropriate position for next phase to start
    }else if(pass_num==1){

    }else{//pass_num==2 case
        int array_start_idx=d_prefix_indices_pass_num[matrix_k];
        int* array = &prefix_global[array_start_idx];
        //just inclusive belloch for this array
        int array_size=numThreadsPerBlock;//need to keep it power of 2,fixed as 1024
        
        //propgate it in next kernel
    }
    if (k >= numMatrices) return;
    int maxV = d_range[k];
    int rows = d_rows[k];
    int cols = d_cols[k];
    //int totalElements = rows * cols;
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
    // int offset = 0;
    // for (int m = 0; m < k; m++) {
    //     offset += d_rows[m] * d_cols[m];
    // }
    // int* matrix = d_input + offset;
    // for (int val = tid; val <= maxV; val += blockDim.x) {
    //     int start = prefixSumArray[val];
    //     int end = (val == maxV) ? totalElements : prefixSumArray[val + 1];
    //     if (start >= totalElements || end > totalElements) continue;
    //     for (int pos = start; pos < end; pos++) {
    //         matrix[pos] = val;
    //     }
    // }
}

__global__ void writeBackKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, const int* d_prefix_blocks_write, int max_padded_size, int numMatrices, int* prefix_global) {
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
    int elements = d_rows[matrix_k] * d_cols[matrix_k];
    int* prefixArr = &prefix_global[matrix_k * max_padded_size];

    // for (int m = 0; m < matrix_k; m++) {
    //     offset += d_rows[m] * d_cols[m];
    // }
    //int* matrix = d_input + offset;
    //int blocks_before = block_idx - d_prefix_blocks[matrix_k];
    //int start = blocks_before * threadsPerBlock;
    //int end = min(start + threadsPerBlock, elements);
    // for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
    //     int val = matrix[i];
    //     if (val <= maxV) {
    //         atomicAdd(&freqArray[val], 1);
    //     }
    // }
    
    int block_offset = block_idx - d_prefix_blocks_write[matrix_k];
    int val_start = block_offset * blockDim.x;
    int val_end = min(val_start + blockDim.x, maxV + 1);
    
    int input_offset = 0;
    for (int m = 0; m < matrix_k; m++){
        input_offset += d_rows[m] * d_cols[m];
    }
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
    vector<int> rows, cols, prefix_indices_pass_0, prefix_indices_pass_1, prefix_indices_pass_2,prefix_blocks_pass_0,prefix_blocks_pass_1,prefix_blocks_pass_2;
    int *d_input = nullptr, *d_range = nullptr;
    int *d_rows = nullptr;
    int *d_cols = nullptr;
    int *d_prefix_indices_pass_0 = nullptr, *d_prefix_indices_pass_1 = nullptr, *d_prefix_indices_pass_2 = nullptr;
    int *prefix_global = nullptr;
    int*d_prefix_blocks_pass_0 = nullptr,*d_prefix_blocks_pass_1 = nullptr,*d_prefix_blocks_pass_2 = nullptr;
    int numThreads = 1024;
    try {
        const int numMatrices = matrices.size();
        rows.resize(numMatrices);
        cols.resize(numMatrices);
        prefix_indices_pass_0.resize(numMatrices);
        prefix_indices_pass_1.resize(numMatrices);
        prefix_indices_pass_2.resize(numMatrices);
        prefix_blocks_pass_0.resize(numMatrices+1);
        prefix_blocks_pass_1.resize(numMatrices+1);
        prefix_blocks_pass_2.resize(numMatrices+1);
        int Elements_0 = 0;
        int Elements_1 = 0;
        int Elements_2 = 0;
        int Blocks_0 = 0;
        int Blocks_1 = 0;
        int Blocks_2 = 0;
        int totalElements = 0;//total elements in the input
        int totalElementsInPrefix = 0;
        for (int i = 0; i < numMatrices; i++) {
            if (matrices[i].empty() || matrices[i][0].empty()) {
                throw runtime_error("Empty matrix detected");
            }
            rows[i] = matrices[i].size();
            cols[i] = matrices[i][0].size();
            prefix_indices_pass_0[i] = Elements_0+Elements_1+Elements_2;
            if(i>0){
                prefix_indices_pass_0[i]+=prefix_indices_pass_0[i-1];
            }
            Elements_0 = rows[i] * cols[i];
            Blocks_0 = static_cast<int>(std::ceil(static_cast<double>(Elements_0) / numThreads));
            prefix_blocks_pass_0[i+1]=prefix_blocks_pass_0[i]+Blocks_0;
            prefix_indices_pass_1[i] = Elements_0+Elements_1+Elements_2;
            if(i>0){
                prefix_indices_pass_1[i]+=prefix_indices_pass_1[i-1];
            }
            Elements_1 = Blocks_0*numThreads;
            Blocks_1 = static_cast<int>(std::ceil(static_cast<double>(Elements_1) / numThreads));
            prefix_blocks_pass_1[i+1]=prefix_blocks_pass_1[i]+Blocks_1;
            prefix_indices_pass_2[i] = Elements_0+Elements_1+Elements_2;
            if(i>0){
                prefix_indices_pass_2[i]+=prefix_indices_pass_2[i-1];
            }
            Elements_2 = Blocks_1*numThreads;
            Blocks_1 = static_cast<int>(std::ceil(static_cast<double>(Elements_2) / numThreads));
            prefix_blocks_pass_2[i+1]=prefix_blocks_pass_2[i]+Blocks_2;
            totalElementsInPrefix+=Elements_0+Elements_1+Elements_2;
            totalElements+=1;
        }
        int pos=0;
        host_input = new int[totalElements];
        for (int k = 0; k < numMatrices; k++) {
            for (const auto& row : matrices[k]) {
                for (int val : row) {
                    host_input[pos++] = val;
                }
            }
        }
        checkCuda(cudaMalloc(&d_input, totalElements * sizeof(int)), "d_input alloc");
        CudaPtrGuard guard_d_input(reinterpret_cast<void**>(&d_input));
        checkCuda(cudaMalloc(&d_range, numMatrices * sizeof(int)), "d_range alloc");
        CudaPtrGuard guard_d_range(reinterpret_cast<void**>(&d_range));
        checkCuda(cudaMalloc(&d_rows, numMatrices * sizeof(int)), "d_rows alloc");
        CudaPtrGuard guard_d_rows(reinterpret_cast<void**>(&d_rows));
        checkCuda(cudaMalloc(&d_cols, numMatrices * sizeof(int)), "d_cols alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_cols));
        checkCuda(cudaMalloc(&d_prefix_indices_pass_0, numMatrices * sizeof(int)), "d_prefix_indices_pass_0 alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_prefix_indices_pass_0));
        checkCuda(cudaMalloc(&d_prefix_indices_pass_1, numMatrices * sizeof(int)), "d_prefix_indices_pass_1 alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_prefix_indices_pass_1));
        checkCuda(cudaMalloc(&d_prefix_indices_pass_2, numMatrices * sizeof(int)), "d_prefix_indices_pass_2 alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_prefix_indices_pass_2));
        checkCuda(cudaMalloc(&d_prefix_blocks_pass_0, numMatrices * sizeof(int)), "d_prefix_blocks_pass_0 alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_prefix_blocks_pass_0));
        checkCuda(cudaMalloc(&d_prefix_blocks_pass_1, numMatrices * sizeof(int)), "d_prefix_blocks_pass_1 alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_prefix_blocks_pass_1));
        checkCuda(cudaMalloc(&d_prefix_blocks_pass_2, numMatrices * sizeof(int)), "d_prefix_blocks_pass_2 alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_prefix_blocks_pass_2));
        checkCuda(cudaMalloc(&prefix_global, totalElementsInPrefix * sizeof(int)),"prefix_global alloc");
        CudaPtrGuard guard_prefix_global(reinterpret_cast<void**>(&prefix_global));
        checkCuda(cudaMemcpy(d_input, host_input, totalElements * sizeof(int), cudaMemcpyHostToDevice), "d_input copy");
        checkCuda(cudaMemcpy(d_range, range.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_range copy");
        checkCuda(cudaMemcpy(d_rows, rows.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_rows copy");
        checkCuda(cudaMemcpy(d_cols, cols.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_cols copy");
        checkCuda(cudaMemcpy(d_prefix_indices_pass_0, prefix_indices_pass_0.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_indices_pass_0 copy");
        checkCuda(cudaMemcpy(d_prefix_indices_pass_1, prefix_indices_pass_1.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_indices_pass_1 copy");
        checkCuda(cudaMemcpy(d_prefix_indices_pass_2, prefix_indices_pass_2.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_indices_pass_2 copy");
        checkCuda(cudaMemcpy(d_prefix_blocks_pass_0, prefix_blocks_pass_0.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_blocks_pass_0 copy");
        checkCuda(cudaMemcpy(d_prefix_blocks_pass_1, prefix_blocks_pass_1.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_blocks_pass_1 copy");
        checkCuda(cudaMemcpy(d_prefix_blocks_pass_2, prefix_blocks_pass_2.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_blocks_pass_2 copy");

        // initFreqKernel<<<numMatrices, 1024>>>(prefix_global, d_range, max_padded_size, numMatrices);
        // checkCuda(cudaGetLastError(), "initFreqKernel launch");
        // checkCuda(cudaDeviceSynchronize(), "initFreqKernel sync");
        countFreqKernel<<<prefix_blocks[numMatrices], 1024>>>(d_input, d_range, d_rows, d_cols, d_prefix_blocks, numMatrices, prefix_global, max_padded_size, 1024);
        checkCuda(cudaGetLastError(), "countFreqKernel launch");
        checkCuda(cudaDeviceSynchronize(), "countFreqKernel sync");
        preFixSumKernel<<<numMatrices, 1024>>>(d_input, d_range, d_rows, d_cols, numMatrices, prefix_global, max_padded_size);
        checkCuda(cudaGetLastError(), "preFixSumKernel launch");
        checkCuda(cudaDeviceSynchronize(), "preFixSumKernel sync");
        vector<int> prefix_blocks_write(numMatrices + 1, 0);
        for (int i = 0; i < numMatrices; ++i) {
            int blocks = (range[i] + 1 + 1023) / 1024;
            prefix_blocks_write[i+1] = prefix_blocks_write[i] + blocks;
        }
        checkCuda(cudaMalloc(&d_prefix_blocks_write, (numMatrices+1)*sizeof(int)), "d_prefix_blocks_write alloc");
        checkCuda(cudaMemcpy(d_prefix_blocks_write, prefix_blocks_write.data(), (numMatrices+1)*sizeof(int), cudaMemcpyHostToDevice), "d_prefix_blocks_write copy");
        writeBackKernel<<<prefix_blocks_write[numMatrices], 1024>>>(d_input, d_range, d_rows, d_cols, d_prefix_blocks_write, max_padded_size, numMatrices, prefix_global);
        checkCuda(cudaGetLastError(), "writeBackSumKernel launch");
        checkCuda(cudaDeviceSynchronize(), "writeBackSumKernel sync");
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
        host_input = nullptr;
        return matrices;
    } catch (...) {
        if (host_input) delete[] host_input;
        if (d_input) cudaFree(d_input);
        if (d_range) cudaFree(d_range);
        if (d_rows) cudaFree(d_rows);
        if (d_cols) cudaFree(d_cols);
        if (prefix_global) cudaFree(prefix_global);
        if (d_prefix_blocks) cudaFree(d_prefix_blocks);
        cudaDeviceReset();
        throw;
    }
}