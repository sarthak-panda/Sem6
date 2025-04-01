#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
using namespace std;
int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}
__global__ void countFreqKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, const int* d_prefix_blocks_count_freq, int* prefix_global, int numMatrices, int threadsPerBlock,const int*d_prefix_indices_pass_0) {
    int block_idx = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_count_freq[i] <= block_idx && block_idx < d_prefix_blocks_count_freq[i+1]) {
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
    int blocks_before = block_idx - d_prefix_blocks_count_freq[matrix_k];
    int start = blocks_before * threadsPerBlock;
    int i = start + threadIdx.x;
    if (i >= elements) return;
    int val = matrix[i];
    if (val <= maxV) {
        atomicAdd(&freqArray[val], 1);
    }
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
__global__ void preFixSumKernel(int* prefix_global, const int* d_prefix_blocks_pass_num, const int* d_prefix_indices_pass_num, const int* d_prefix_indices_pass_num_next, int numThreadsPerBlock, int numMatrices, int pass_num) {
    int k = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_pass_num[i] <= k && k < d_prefix_blocks_pass_num[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;
    int blocks_before = k - d_prefix_blocks_pass_num[matrix_k];
    int array_start_idx=d_prefix_indices_pass_num[matrix_k]+blocks_before*numThreadsPerBlock;
    int* array = &prefix_global[array_start_idx];
    int array_size=numThreadsPerBlock;
    inclusiveBlellochScan(array, array_size);
    if(pass_num==0||pass_num==1){
        int StartIdx_Pass_next_MatK = d_prefix_indices_pass_num_next[matrix_k];
        int blocks_before = k - d_prefix_blocks_pass_num[matrix_k];
        int ValueToWrite = array[array_size-1];
        int locationToWrite = StartIdx_Pass_next_MatK+blocks_before;
        prefix_global[locationToWrite] = ValueToWrite;
    }
}
__global__ void resolveHierarchyKernel(int* prefix_global, const int* d_prefix_blocks_pass_num, const int* d_prefix_indices_pass_num, const int* d_prefix_indices_pass_num_next, int numMatrices,int hierarchy_num){
    int k = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_pass_num[i] <= k && k < d_prefix_blocks_pass_num[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;
    int blocks_before = k - d_prefix_blocks_pass_num[matrix_k];
    int array_start_idx=d_prefix_indices_pass_num[matrix_k]+blocks_before*blockDim.x;//numThreadsPerBlock;
    int* array = &prefix_global[array_start_idx];
    //int array_size=numThreadsPerBlock;
    //int array_size=blockDim.x;
    if(hierarchy_num==1||hierarchy_num==0){
        int StartIdx_Pass_next_MatK = d_prefix_indices_pass_num_next[matrix_k];
        int blocks_before = k - d_prefix_blocks_pass_num[matrix_k];
        if(blocks_before==0) return;
        int locationToRead = StartIdx_Pass_next_MatK+blocks_before-1;
        int data = prefix_global[locationToRead];
        int tid=threadIdx.x;
        array[tid]+=data;
    }
}
__global__ void writeBackKernel(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, const int* d_prefix_blocks_pass_0,const int* d_prefix_indices_pass_0, int numMatrices, int* prefix_global, int numThreadsPerBlock){
    int k = blockIdx.x;
    int matrix_k = -1;
    for (int i = 0; i < numMatrices; ++i) {
        if (d_prefix_blocks_pass_0[i] <= k && k < d_prefix_blocks_pass_0[i+1]) {
            matrix_k = i;
            break;
        }
    }
    if (matrix_k == -1) return;
    int maxV = d_range[matrix_k];
    int elements = d_rows[matrix_k] * d_cols[matrix_k];
    int startIndexOfFreqArray = d_prefix_indices_pass_0[matrix_k];
    int* preFixArray = &prefix_global[startIndexOfFreqArray];
    int input_offset = 0;
    for (int m = 0; m < matrix_k; m++){
        input_offset += d_rows[m] * d_cols[m];
    }
    int block_offset = k - d_prefix_blocks_pass_0[matrix_k];
    int val_start = block_offset * numThreadsPerBlock;
    int val_end = min(val_start + numThreadsPerBlock, maxV+1);
    for (int val = val_start + threadIdx.x; val < val_end; val += blockDim.x) {
        int start = 0;
        if(val>=1){
            start = preFixArray[val-1];
        }
        int end = (val == maxV) ? elements : preFixArray[val];
        int* matrix = d_input + input_offset;
        for (int pos = start; pos < end; ++pos) matrix[pos] = val;
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
    vector<int> rows, cols,prefix_indices_count_freq, prefix_indices_pass_0, prefix_indices_pass_1, prefix_indices_pass_2,prefix_blocks_count_freq,prefix_blocks_pass_0,prefix_blocks_pass_1,prefix_blocks_pass_2;
    int *d_input = nullptr, *d_range = nullptr;
    int *d_rows = nullptr;
    int *d_cols = nullptr;
    int *d_prefix_indices_pass_0 = nullptr, *d_prefix_indices_pass_1 = nullptr, *d_prefix_indices_pass_2 = nullptr;
    int *prefix_global = nullptr;
    int*d_prefix_blocks_pass_0 = nullptr,*d_prefix_blocks_pass_1 = nullptr,*d_prefix_blocks_pass_2 = nullptr;
    int*d_prefix_indices_count_freq=nullptr,*d_prefix_blocks_count_freq=nullptr;
    vector<int> h_p_g;
    int numThreads = 1024;
    try {
        const int numMatrices = matrices.size();
        rows.resize(numMatrices);
        cols.resize(numMatrices);
        prefix_indices_count_freq.resize(numMatrices);
        prefix_indices_pass_0.resize(numMatrices);
        prefix_indices_pass_1.resize(numMatrices);
        prefix_indices_pass_2.resize(numMatrices);
        prefix_blocks_pass_0.resize(numMatrices+1);
        prefix_blocks_pass_1.resize(numMatrices+1);
        prefix_blocks_pass_2.resize(numMatrices+1);
        prefix_blocks_count_freq.resize(numMatrices+1);
        int Mat_Elements = 0;
        int Mat_Blocks = 0;
        int totalCountFreqBlocks = 0;
        int Elements_0 = 0;
        int Elements_1 = 0;
        int Elements_2 = 0;
        int Blocks_0 = 0;
        int Blocks_1 = 0;
        int Blocks_2 = 0;
        int TotalBlocks_0 = 0;
        int TotalBlocks_1 = 0;
        int TotalBlocks_2 = 0;
        int totalElements = 0;
        int totalElementsInPrefix = 0;
        for (int i = 0; i < numMatrices; i++) {
            if (matrices[i].empty() || matrices[i][0].empty()) {
                throw runtime_error("Empty matrix detected");
            }
            rows[i] = matrices[i].size();
            cols[i] = matrices[i][0].size();
            // prefix_indices_pass_0[i] = Elements_0+Elements_1+Elements_2;
            // if(i>0){
            //     prefix_indices_pass_0[i]+=prefix_indices_pass_0[i-1];
            // }
            Elements_0 = range[i]+1;
            Blocks_0 = static_cast<int>(std::ceil(static_cast<double>(Elements_0) / numThreads));
            Elements_0 = Blocks_0*numThreads;
            TotalBlocks_0+=Blocks_0;
            prefix_blocks_pass_0[i+1]=prefix_blocks_pass_0[i]+Blocks_0;
            // prefix_indices_pass_1[i] = Elements_0+Elements_1+Elements_2;
            // if(i>0){
            //     prefix_indices_pass_1[i]+=prefix_indices_pass_1[i-1];
            // }
            Elements_1 = Blocks_0;
            Blocks_1 = static_cast<int>(std::ceil(static_cast<double>(Elements_1) / numThreads));
            Elements_1 = Blocks_1*numThreads;
            TotalBlocks_1+=Blocks_1;
            prefix_blocks_pass_1[i+1]=prefix_blocks_pass_1[i]+Blocks_1;
            // prefix_indices_pass_2[i] = Elements_0+Elements_1+Elements_2;
            // if(i>0){
            //     prefix_indices_pass_2[i]+=prefix_indices_pass_2[i-1];
            // }
            Elements_2 = Blocks_1;
            Blocks_2 = static_cast<int>(std::ceil(static_cast<double>(Elements_2) / numThreads));
            Elements_2 = Blocks_2*numThreads;
            TotalBlocks_2+=Blocks_2;
            prefix_blocks_pass_2[i+1]=prefix_blocks_pass_2[i]+Blocks_2;

            prefix_indices_pass_0[i] = totalElementsInPrefix;
            prefix_indices_pass_1[i] = totalElementsInPrefix + Elements_0;
            prefix_indices_pass_2[i] = totalElementsInPrefix + Elements_0 + Elements_1;

            totalElementsInPrefix+=Elements_0+Elements_1+Elements_2;
            Mat_Elements=rows[i] * cols[i];
            Mat_Blocks = static_cast<int>(std::ceil(static_cast<double>(Mat_Elements) / numThreads));
            prefix_indices_count_freq[i]=totalElements;
            prefix_blocks_count_freq[i+1]=prefix_blocks_count_freq[i]+Mat_Blocks;
            totalElements+=Mat_Elements;
            totalCountFreqBlocks+=Mat_Blocks;
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

        h_p_g.resize(totalElementsInPrefix);


        checkCuda(cudaMalloc(&d_input, totalElements * sizeof(int)), "d_input alloc");
        CudaPtrGuard guard_d_input(reinterpret_cast<void**>(&d_input));
        checkCuda(cudaMalloc(&d_range, numMatrices * sizeof(int)), "d_range alloc");
        CudaPtrGuard guard_d_range(reinterpret_cast<void**>(&d_range));
        checkCuda(cudaMalloc(&d_rows, numMatrices * sizeof(int)), "d_rows alloc");
        CudaPtrGuard guard_d_rows(reinterpret_cast<void**>(&d_rows));
        checkCuda(cudaMalloc(&d_cols, numMatrices * sizeof(int)), "d_cols alloc");
        CudaPtrGuard guard_d_cols(reinterpret_cast<void**>(&d_cols));
        checkCuda(cudaMalloc(&d_prefix_indices_count_freq, numMatrices * sizeof(int)), "d_prefix_indices_count_freq alloc");
        CudaPtrGuard guard_d_prefix_indices_count_freq(reinterpret_cast<void**>(&d_prefix_indices_count_freq));
        checkCuda(cudaMalloc(&d_prefix_indices_pass_0, numMatrices * sizeof(int)), "d_prefix_indices_pass_0 alloc");
        CudaPtrGuard guard_d_prefix_indices_pass_0(reinterpret_cast<void**>(&d_prefix_indices_pass_0));
        checkCuda(cudaMalloc(&d_prefix_indices_pass_1, numMatrices * sizeof(int)), "d_prefix_indices_pass_1 alloc");
        CudaPtrGuard guard_d_prefix_indices_pass_1(reinterpret_cast<void**>(&d_prefix_indices_pass_1));
        checkCuda(cudaMalloc(&d_prefix_indices_pass_2, numMatrices * sizeof(int)), "d_prefix_indices_pass_2 alloc");
        CudaPtrGuard guard_d_prefix_indices_pass_2(reinterpret_cast<void**>(&d_prefix_indices_pass_2));
        checkCuda(cudaMalloc(&d_prefix_blocks_count_freq, (numMatrices+1) * sizeof(int)), "d_prefix_blocks_count_freq alloc");
        CudaPtrGuard guard_d_prefix_blocks_count_freq(reinterpret_cast<void**>(&d_prefix_blocks_count_freq));
        checkCuda(cudaMalloc(&d_prefix_blocks_pass_0, (numMatrices+1) * sizeof(int)), "d_prefix_blocks_pass_0 alloc");
        CudaPtrGuard guard_d_prefix_blocks_pass_0(reinterpret_cast<void**>(&d_prefix_blocks_pass_0));
        checkCuda(cudaMalloc(&d_prefix_blocks_pass_1, (numMatrices+1) * sizeof(int)), "d_prefix_blocks_pass_1 alloc");
        CudaPtrGuard guard_d_prefix_blocks_pass_1(reinterpret_cast<void**>(&d_prefix_blocks_pass_1));
        checkCuda(cudaMalloc(&d_prefix_blocks_pass_2, (numMatrices+1) * sizeof(int)), "d_prefix_blocks_pass_2 alloc");
        CudaPtrGuard guard_d_prefix_blocks_pass_2(reinterpret_cast<void**>(&d_prefix_blocks_pass_2));
        checkCuda(cudaMalloc(&prefix_global, totalElementsInPrefix * sizeof(int)), "prefix_global alloc");
        CudaPtrGuard guard_prefix_global(reinterpret_cast<void**>(&prefix_global));
        checkCuda(cudaMemset(prefix_global, 0, totalElementsInPrefix * sizeof(int)), "memset prefix_global");
        checkCuda(cudaMemcpy(d_input, host_input, totalElements * sizeof(int), cudaMemcpyHostToDevice), "d_input copy");
        checkCuda(cudaMemcpy(d_range, range.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_range copy");
        checkCuda(cudaMemcpy(d_rows, rows.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_rows copy");
        checkCuda(cudaMemcpy(d_cols, cols.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_cols copy");
        checkCuda(cudaMemcpy(d_prefix_indices_count_freq, prefix_indices_count_freq.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_indices_count_freq copy");
        checkCuda(cudaMemcpy(d_prefix_indices_pass_0, prefix_indices_pass_0.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_indices_pass_0 copy");
        checkCuda(cudaMemcpy(d_prefix_indices_pass_1, prefix_indices_pass_1.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_indices_pass_1 copy");
        checkCuda(cudaMemcpy(d_prefix_indices_pass_2, prefix_indices_pass_2.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_indices_pass_2 copy");
        checkCuda(cudaMemcpy(d_prefix_blocks_count_freq, prefix_blocks_count_freq.data(), (numMatrices+1) * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_blocks_count_freq copy");
        checkCuda(cudaMemcpy(d_prefix_blocks_pass_0, prefix_blocks_pass_0.data(), (numMatrices+1) * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_blocks_pass_0 copy");
        checkCuda(cudaMemcpy(d_prefix_blocks_pass_1, prefix_blocks_pass_1.data(), (numMatrices+1) * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_blocks_pass_1 copy");
        checkCuda(cudaMemcpy(d_prefix_blocks_pass_2, prefix_blocks_pass_2.data(), (numMatrices+1) * sizeof(int), cudaMemcpyHostToDevice), "d_prefix_blocks_pass_2 copy");
        countFreqKernel<<<totalCountFreqBlocks,numThreads>>>(d_input, d_range, d_rows, d_cols, d_prefix_blocks_count_freq, prefix_global, numMatrices, numThreads,d_prefix_indices_pass_0);
        checkCuda(cudaGetLastError(), "countFreqKernel launch");
        checkCuda(cudaDeviceSynchronize(), "countFreqKernel sync");

        checkCuda(cudaMemcpy(h_p_g.data(),prefix_global, totalElementsInPrefix * sizeof(int), cudaMemcpyDeviceToHost), "prefix copy");
        for(int i=0;i<totalElementsInPrefix;i++){
            cout<<h_p_g[i]<<" ";
        }
        cout<<endl;
        cout<<"----------------------------"<<endl;

        preFixSumKernel<<<TotalBlocks_0,numThreads>>>(prefix_global,d_prefix_blocks_pass_0,d_prefix_indices_pass_0,d_prefix_indices_pass_1,numThreads,numMatrices,0);
        checkCuda(cudaGetLastError(), "preFixSumKernel_0 launch");
        checkCuda(cudaDeviceSynchronize(), "preFixSumKernel_0 sync");

        checkCuda(cudaMemcpy(h_p_g.data(),prefix_global, totalElementsInPrefix * sizeof(int), cudaMemcpyDeviceToHost), "prefix copy");
        for(int i=0;i<totalElementsInPrefix;i++){
            cout<<h_p_g[i]<<" ";
        }
        cout<<endl;
        cout<<"----------------------------"<<endl;

        preFixSumKernel<<<TotalBlocks_1,numThreads>>>(prefix_global,d_prefix_blocks_pass_1,d_prefix_indices_pass_1,d_prefix_indices_pass_2,numThreads,numMatrices,1);
        checkCuda(cudaGetLastError(), "preFixSumKernel_1 launch");
        checkCuda(cudaDeviceSynchronize(), "preFixSumKernel_1 sync");

        checkCuda(cudaMemcpy(h_p_g.data(),prefix_global, totalElementsInPrefix * sizeof(int), cudaMemcpyDeviceToHost), "prefix copy");
        for(int i=0;i<totalElementsInPrefix;i++){
            cout<<h_p_g[i]<<" ";
        }
        cout<<endl;
        cout<<"----------------------------"<<endl;

        preFixSumKernel<<<TotalBlocks_2,numThreads>>>(prefix_global,d_prefix_blocks_pass_2,d_prefix_indices_pass_2,nullptr,numThreads,numMatrices,2);
        checkCuda(cudaGetLastError(), "preFixSumKernel_2 launch");
        checkCuda(cudaDeviceSynchronize(), "preFixSumKernel_2 sync");

        checkCuda(cudaMemcpy(h_p_g.data(),prefix_global, totalElementsInPrefix * sizeof(int), cudaMemcpyDeviceToHost), "prefix copy");
        for(int i=0;i<totalElementsInPrefix;i++){
            cout<<h_p_g[i]<<" ";
        }
        cout<<endl;
        cout<<"----------------------------"<<endl;

        resolveHierarchyKernel<<<TotalBlocks_1,numThreads>>>(prefix_global,d_prefix_blocks_pass_1,d_prefix_indices_pass_1,d_prefix_indices_pass_2,numMatrices,1);
        checkCuda(cudaGetLastError(), "resolveHierarchyKernel_1 launch");
        checkCuda(cudaDeviceSynchronize(), "resolveHierarchyKernel_1 sync");

        checkCuda(cudaMemcpy(h_p_g.data(),prefix_global, totalElementsInPrefix * sizeof(int), cudaMemcpyDeviceToHost), "prefix copy");
        for(int i=0;i<totalElementsInPrefix;i++){
            cout<<h_p_g[i]<<" ";
        }
        cout<<endl;
        cout<<"----------------------------"<<endl;

        resolveHierarchyKernel<<<TotalBlocks_0,numThreads>>>(prefix_global,d_prefix_blocks_pass_0,d_prefix_indices_pass_0,d_prefix_indices_pass_1,numMatrices,0);
        checkCuda(cudaGetLastError(), "resolveHierarchyKernel_0 launch");
        checkCuda(cudaDeviceSynchronize(), "resolveHierarchyKernel_0 sync");

        checkCuda(cudaMemcpy(h_p_g.data(),prefix_global, totalElementsInPrefix * sizeof(int), cudaMemcpyDeviceToHost), "prefix copy");
        for(int i=0;i<totalElementsInPrefix;i++){
            cout<<h_p_g[i]<<" ";
        }
        cout<<endl;
        cout<<"----------------------------"<<endl;

        writeBackKernel<<<TotalBlocks_0,numThreads>>>(d_input, d_range, d_rows, d_cols,d_prefix_blocks_pass_0,d_prefix_indices_pass_0,numMatrices,prefix_global,numThreads);
        checkCuda(cudaGetLastError(), "writeBackKernel launch");
        checkCuda(cudaDeviceSynchronize(), "writeBackKernel sync");
        checkCuda(cudaMemcpy(host_input, d_input, totalElements * sizeof(int), cudaMemcpyDeviceToHost), "results copy");

        checkCuda(cudaMemcpy(h_p_g.data(),prefix_global, totalElementsInPrefix * sizeof(int), cudaMemcpyDeviceToHost), "prefix copy");
        for(int i=0;i<totalElementsInPrefix;i++){
            cout<<h_p_g[i]<<" ";
        }
        cout<<endl;
        cout<<"----------------------------"<<endl;

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
        if (d_prefix_indices_count_freq) cudaFree(d_prefix_indices_count_freq);
        if (d_prefix_indices_pass_0) cudaFree(d_prefix_indices_pass_0);
        if (d_prefix_indices_pass_1) cudaFree(d_prefix_indices_pass_1);
        if (d_prefix_indices_pass_2) cudaFree(d_prefix_indices_pass_2);
        if (d_prefix_blocks_count_freq) cudaFree(d_prefix_blocks_count_freq);
        if (d_prefix_blocks_pass_0) cudaFree(d_prefix_blocks_pass_0);
        if (d_prefix_blocks_pass_1) cudaFree(d_prefix_blocks_pass_1);
        if (d_prefix_blocks_pass_2) cudaFree(d_prefix_blocks_pass_2);
        cudaDeviceReset();
        throw;
    }
}
