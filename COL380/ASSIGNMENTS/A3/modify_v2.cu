#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include "modify.cuh"

using namespace std;
__global__ void MyKernelFunc(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, int numMatrices){
	int k = blockIdx.x;
    if (k >= numMatrices) return;
	int maxV = d_range[k];
    int rows = d_rows[k];
    int cols = d_cols[k];
    int tid = threadIdx.x;
    int totalElements = rows * cols;
    int n = maxV + 1;
    int padded_n = 1;
    while (padded_n < n) {
        padded_n *= 2;
    }
	int offset = 0;
    for (int m = 0; m < k; m++) {
        offset += d_rows[m] * d_cols[m];
    }
	extern __shared__ int shared[];
    int* freqArray = &shared[0];
    int* prefixSumArray = &shared[n]; 
	int* matrix = d_input+offset;
    for (int i = tid; i <= maxV; i += blockDim.x) {
        freqArray[i] = 0;
    }
    __syncthreads();
    for (int i = tid; i < totalElements; i += blockDim.x) {
        int val = matrix[i];
        if (val <= maxV) {
            atomicAdd(&freqArray[val], 1); 
        }
    }
    __syncthreads();
    for (int i = tid; i < padded_n; i += blockDim.x) {
        if (i < n) {
            prefixSumArray[i] = freqArray[i];
        } else {
            prefixSumArray[i] = 0; 
        }
    }
    __syncthreads();
    for (int stride = 1; stride < padded_n; stride *= 2) {
        for (int i = threadIdx.x; i < padded_n / (2 * stride); i += blockDim.x) {
            int idx = (i + 1) * 2 * stride - 1;
            if (idx < padded_n) {
                prefixSumArray[idx] += prefixSumArray[idx - stride];
            }
        }
        __syncthreads();
    }
    if(tid==0)prefixSumArray[padded_n - 1] = 0;
    __syncthreads();
    for (int stride = padded_n / 2; stride > 0; stride /= 2) {
        for (int i = threadIdx.x; i < padded_n / (2 * stride); i += blockDim.x) {
            int idx = (i + 1) * 2 * stride - 1;
            if (idx < padded_n) {
                int temp = prefixSumArray[idx - stride];
                prefixSumArray[idx - stride] = prefixSumArray[idx];
                prefixSumArray[idx] += temp;
            }
        }
        __syncthreads();
    }
	for (int val = threadIdx.x; val <= maxV; val += blockDim.x) {
        int start = prefixSumArray[val];
        int end = (val == maxV) ? totalElements : prefixSumArray[val + 1];
        for (int pos = start; pos < end; pos++) {
            matrix[pos] = val;
        }
    }
}
vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& range){
	int numMatrices = matrices.size();
	int totalElements = 0;
    vector<int> rows(numMatrices), cols(numMatrices);
    for (int i = 0; i < numMatrices; i++) {
        int r = matrices[i].size();
        int c = matrices[i][0].size();
        rows[i] = r;
        cols[i] = c;
        totalElements += r * c;
    }
	int* input = new int[totalElements];
    int pos = 0;
    for (int k = 0; k < numMatrices; k++) {
        for (int i = 0; i < rows[k]; i++) {
            for (int j = 0; j < cols[k]; j++) {
                input[pos++] = matrices[k][i][j];
            }
        }
    }
	int *device_input, *device_range, *device_rows, *device_cols;
	cudaMalloc(&device_input, totalElements * sizeof(int));
    cudaMalloc(&device_range, numMatrices * sizeof(int));
    cudaMalloc(&device_rows, numMatrices * sizeof(int));
    cudaMalloc(&device_cols, numMatrices * sizeof(int));
	cudaMemcpy(device_input, input, totalElements * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_range, range.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_rows, rows.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_cols, cols.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);
	int threadsPerBlock = 256;
    int numBlocks = numMatrices;
	int maxGlobal = 0;
    for (int v : range) {
        if (v > maxGlobal) maxGlobal = v;
    }
    int nextTwoPowerMaxGlobal = 1;
    while(nextTwoPowerMaxGlobal < (maxGlobal+1)) nextTwoPowerMaxGlobal *= 2;
    cout<<maxGlobal<<" "<<nextTwoPowerMaxGlobal<<endl;
    int sharedSize = (maxGlobal + 1) * sizeof(int);
    sharedSize += (nextTwoPowerMaxGlobal) * sizeof(int);
	MyKernelFunc<<<numBlocks, threadsPerBlock, sharedSize>>>(device_input, device_range, device_rows, device_cols, numMatrices);
	cudaDeviceSynchronize();
	cudaMemcpy(input, device_input, totalElements * sizeof(int), cudaMemcpyDeviceToHost);
	pos = 0;
    for (int k = 0; k < numMatrices; k++) {
        int r = rows[k];
        int c = cols[k];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                matrices[k][i][j] = input[pos++];
            }
        }
    }
	cudaFree(device_input);
    cudaFree(device_range);
    cudaFree(device_rows);
    cudaFree(device_cols);
    delete[] input;
	return matrices;
}