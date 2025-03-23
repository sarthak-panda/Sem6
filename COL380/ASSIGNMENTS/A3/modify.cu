#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
using namespace std;

__global__ void MyKernelFunc(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, int numMatrices){
	int k = blockIdx.x;
    if (k >= numMatrices) return;
	int maxV = d_range[k];
    int rows = d_rows[k];
    int cols = d_cols[k];
    int totalElements = rows * cols;
	int offset = 0;
    for (int m = 0; m < k; m++) {
        offset += d_rows[m] * d_cols[m];
    }
	extern __shared__ int shared[];
    int* freqArray = shared;
    int* prefixSumArray = &shared[maxV + 1]; 
	int* matrix = d_input+offset;
	for (int i = 0; i <= maxV; ++i) {
        freqArray[i] = 0;
    }
    for (int i = 0; i < totalElements; ++i) {
        int val = matrix[i];
        if (val <= maxV) {
            freqArray[val]++;
        }
    }
	prefixSumArray[0] = 0;
    for (int i = 1; i <= maxV; ++i) {
        prefixSumArray[i] = prefixSumArray[i - 1] + freqArray[i - 1];
    }
	for (int val = 0; val <= maxV; ++val) {
        int count = freqArray[val];
        int start = prefixSumArray[val];
        for (int c = 0; c < count; ++c) {
            int pos = start + c;
            if (pos >= totalElements) break;
            matrix[pos] = val;
        }
    }
	return;
}

vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& range){
	int numMatrices = matrices.size();
	// First, flatten matrices into a single contiguous array.
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

	cudaMemcpy(device_input, input, numMatrices * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_range, range.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_rows, rows.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_cols, cols.data(), numMatrices * sizeof(int), cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
    int numBlocks = numMatrices;
	int maxGlobal = 0;
    for (int v : range) {
        if (v > maxGlobal) maxGlobal = v;
    }
    int sharedSize = 2 * (maxGlobal + 1) * sizeof(int);
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