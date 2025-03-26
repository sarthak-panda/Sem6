#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <iostream>

using namespace std;

// __global__ void MyKernelFunc(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, int numMatrices){
// 	int k = blockIdx.x;
//     if (k >= numMatrices) return;
// 	int maxV = d_range[k];
//     int rows = d_rows[k];
//     int cols = d_cols[k];
//     int tid = threadIdx.x;
//     int totalElements = rows * cols;
//     int n = maxV + 1;
//     int padded_n = 1;
//     while (padded_n < n) {
//         padded_n *= 2;
//     }
// 	int offset = 0;
//     for (int m = 0; m < k; m++) {
//         offset += d_rows[m] * d_cols[m];
//     }
// 	extern __shared__ int shared[];
//     int* freqArray = &shared[0];
//     int* prefixSumArray = &shared[padded_n]; 
// 	int* matrix = d_input+offset;

// 	// for (int i = 0; i <= maxV; ++i) {
//     //     freqArray[i] = 0;
//     // }
//     for (int i = tid; i <= maxV; i += blockDim.x) {
//         freqArray[i] = 0;
//     }
//     __syncthreads();

//     // for (int i = 0; i < totalElements; i++) {
//     //     int val = matrix[i];
//     //     if (val <= maxV) {
//     //         freqArray[val]++;
//     //     }
//     // }
//     for (int i = tid; i < totalElements; i += blockDim.x) {
//         int val = matrix[i];
//         if (val <= maxV) {
//             atomicAdd(&freqArray[val], 1); // Atomic to avoid race
//         }
//     }
//     __syncthreads();
    
//         // prefixSumArray[0] = 0;
//         // for (int i = 1; i <= maxV; i++) {
//         //     prefixSumArray[i] = prefixSumArray[i - 1] + freqArray[i - 1];
//         // }
//     // if (tid == 0) {
//     //     prefixSumArray[0] = 0;
//     //     for (int i = 1; i <= maxV; i++) {
//     //         prefixSumArray[i] = prefixSumArray[i - 1] + freqArray[i - 1];
//     //     }
//     // }
//     // __syncthreads();

//     //Blelloch Scan
//     //prefixSumArray[0] = 0;
//     // for (int i = 1; i <= maxV; i++) {
//     //     prefixSumArray[i] = freqArray[i - 1];
//     // }
//     // __syncthreads();

//     for (int i = tid; i < padded_n; i += blockDim.x) {
//         if (i < n) {
//             prefixSumArray[i] = freqArray[i];
//         } else {
//             prefixSumArray[i] = 0; // Pad with zeros
//         }
//     }
//     __syncthreads();

//     // Up-sweep phase
//     for (int stride = 1; stride < padded_n; stride *= 2) {
//         for (int i = threadIdx.x; i < padded_n / (2 * stride); i += blockDim.x) {
//             int idx = (i + 1) * 2 * stride - 1;
//             if (idx < padded_n) {
//                 prefixSumArray[idx] += prefixSumArray[idx - stride];
//             }
//         }
//         __syncthreads();
//     }

//     // Down-sweep phase
//     prefixSumArray[padded_n - 1] = 0;
//     for (int stride = padded_n / 2; stride > 0; stride /= 2) {
//         for (int i = threadIdx.x; i < padded_n / (2 * stride); i += blockDim.x) {
//             int idx = (i + 1) * 2 * stride - 1;
//             if (idx < padded_n) {
//                 int temp = prefixSumArray[idx - stride];
//                 prefixSumArray[idx - stride] = prefixSumArray[idx];
//                 prefixSumArray[idx] += temp;
//             }
//         }
//         __syncthreads();
//     }

//     // for (int i = tid; i < n; i += blockDim.x) {
//     //     prefixSumArray[i] = (i == 0) ? 0 : prefixSumArray[i - 1];
//     // }
//     // __syncthreads();
//     if (threadIdx.x == 0) {
//         printf("Block %d freqArray: ", blockIdx.x);
//         for (int i = 0; i <= maxV; i++) {
//             printf("%d ", freqArray[i]);
//         }
//         printf("\n");

//         printf("Block %d prefixSumArray: ", blockIdx.x);
//         for (int i = 0; i < padded_n; i++) {
//             printf("%d ", prefixSumArray[i]);
//         }
//         printf("\n");
//     }

// 	for (int val = 0; val <= maxV; val++) {
//         int count = freqArray[val];
//         int start = prefixSumArray[val];
//         for (int c = 0; c < count; ++c) {
//             int pos = start + c;
//             if (pos >= totalElements) break;
//             matrix[pos] = val;
//         }
//     }
// 	//return;
// }

// __global__ void MyKernelFunc(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, int numMatrices) {
//     int k = blockIdx.x;
//     if (k >= numMatrices) return;
//     int maxV = d_range[k];
//     int rows = d_rows[k];
//     int cols = d_cols[k];
//     int tid = threadIdx.x;
//     int totalElements = rows * cols;
//     int n = maxV + 1;

//     // Compute next power of two for padded_n
//     int padded_n = 1;
//     while (padded_n < n) padded_n *= 2;

//     // Calculate offset for this matrix
//     int offset = 0;
//     for (int m = 0; m < k; m++) offset += d_rows[m] * d_cols[m];
    
//     extern __shared__ int shared[];
//     int* freqArray = &shared[0];
//     int* prefixSumArray = &shared[n]; // Start at padded_n

//     // Initialize freqArray to 0 (parallel)
//     for (int i = tid; i <= maxV; i += blockDim.x) freqArray[i] = 0;
//     __syncthreads();

//     // Compute frequencies (parallel with atomicAdd)
//     for (int i = tid; i < totalElements; i += blockDim.x) {
//         int val = d_input[offset + i];
//         if (val <= maxV) atomicAdd(&freqArray[val], 1);
//     }
//     __syncthreads();

//     // Initialize prefixSumArray with freqArray[i-1] (exclusive scan)
//     for (int i = tid; i < padded_n; i += blockDim.x) {
//         prefixSumArray[i] = (i < n) ? (freqArray[i]) : 0;
//     }
//     __syncthreads();

//     // Up-sweep phase of Blelloch scan
//     for (int stride = 1; stride < padded_n; stride *= 2) {
//         for (int i = tid; i < padded_n / (2 * stride); i += blockDim.x) {
//             int idx = (2 * stride) * (i + 1) - 1;
//             if (idx < padded_n) {
//                 prefixSumArray[idx] += prefixSumArray[idx - stride];
//             }
//         }
//         __syncthreads();
//     }

//     // Down-sweep phase
//     if (tid == 0) prefixSumArray[padded_n - 1] = 0;
//     __syncthreads();
//     for (int stride = padded_n / 2; stride >= 1; stride /= 2) {
//         for (int i = tid; i < padded_n / (2 * stride); i += blockDim.x) {
//             int idx = (2 * stride) * (i + 1) - 1;
//             if (idx < padded_n) {
//                 int temp = prefixSumArray[idx - stride];
//                 prefixSumArray[idx - stride] = prefixSumArray[idx];
//                 prefixSumArray[idx] += temp;
//             }
//         }
//         __syncthreads();
//     }

//     if (threadIdx.x == 0) {
//         printf("Block %d freqArray: ", blockIdx.x);
//         for (int i = 0; i <= maxV; i++) {
//             printf("%d ", freqArray[i]);
//         }
//         printf("\n");

//         printf("Block %d prefixSumArray: ", blockIdx.x);
//         for (int i = 0; i < padded_n; i++) {
//             printf("%d ", prefixSumArray[i]);
//         }
//         printf("\n");
//     }

//     // Fill the matrix in sorted order
//     for (int val = 0; val <= maxV; val++) {
//         int count = freqArray[val];
//         int start = prefixSumArray[val];
//         for (int c = 0; c < count; ++c) {
//             int pos = start + c;
//             if (pos < totalElements) d_input[offset + pos] = val;
//         }
//     }
// }

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

    // if (threadIdx.x == 0) {
    //     printf("Block %d freqArray: ", blockIdx.x);
    //     for (int i = 0; i <= maxV; i++) {
    //         printf("%d ", freqArray[i]);
    //     }
    //     printf("\n");

    //     printf("Block %d prefixSumArray: ", blockIdx.x);
    //     for (int i = 0; i < padded_n; i++) {
    //         printf("%d ", prefixSumArray[i]);
    //     }
    //     printf("\n");
    // }

	// for (int val = 0; val <= maxV; val++) {
    //     int count = freqArray[val];
    //     int start = prefixSumArray[val];
    //     for (int c = 0; c < count; ++c) {
    //         int pos = start + c;
    //         if (pos >= totalElements) break;
    //         matrix[pos] = val;
    //     }
    // }

    // for (int val = 0; val <= maxV; val++) {
    //     int start = prefixSumArray[val];
    //     int end = prefixSumArray[val + 1];
    //     for (int pos = start + tid; pos < end; pos += blockDim.x) {
    //         if (pos < totalElements) matrix[pos] = val;
    //     }
    // }
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
                //cout<<input[pos-1]<<"|";
            }
        }
        //cout<<endl;
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
    //int sharedSize = 2 * (maxGlobal + 1) * sizeof(int);
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
                //cout<<input[pos-1]<<"|";
            }
        }
        //cout<<endl;
    }

	cudaFree(device_input);
    cudaFree(device_range);
    cudaFree(device_rows);
    cudaFree(device_cols);
    delete[] input;

	return matrices;
}

int main() {
    // Example usage:
    // A 4x3 matrix as given in the problem statement.
    vector<vector<vector<int>>> matrices = {
    {
        {1, 3, 2},
        {0, 0, 1},
        {3, 2, 0},
        {0, 1, 1}
    },
    {
        {1, 3, 2},
        {4, 6, 5},
        {7, 3, 6},
        {4, 7, 1}
    },
    {
        {1, 3, 2},
        {4, 6, 5},
        {7, 9, 8},
        {9, 7, 1}
    }
    };
    // Upper bound for this matrix's elements is 9.
    vector<int> range = {3,7,9};

    vector<vector<vector<int>>> result = modify(matrices, range);

    // Print the modified matrix.
    for(auto& mat :result){
        for (auto& row : mat) {
            for (auto val : row)
                cout << val << " ";
            cout << "\n";
        }
        cout<<"------\n";
    }
    return 0;
}


////////////////////////////////////////////////////////////////////////

#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <iostream>

using namespace std;

__global__ void MyKernelFunc(int* d_input, const int* d_range, const int* d_rows, const int* d_cols, int numMatrices){
	int k = blockIdx.x;
    if (k >= numMatrices) return;
	int maxV = d_range[k];
    int rows = d_rows[k];
    int cols = d_cols[k];
    int tid = threadIdx.x;
    int totalElements = rows * cols;
	int offset = 0;
    for (int m = 0; m < k; m++) {
        offset += d_rows[m] * d_cols[m];
    }
	extern __shared__ int shared[];
    int* freqArray = &shared[0];
    int* prefixSumArray = &shared[maxV + 1]; 
	int* matrix = d_input+offset;

	// for (int i = 0; i <= maxV; ++i) {
    //     freqArray[i] = 0;
    // }
    for (int i = tid; i <= maxV; i += blockDim.x) {
        freqArray[i] = 0;
    }
    __syncthreads();

    // for (int i = 0; i < totalElements; i++) {
    //     int val = matrix[i];
    //     if (val <= maxV) {
    //         freqArray[val]++;
    //     }
    // }
    for (int i = tid; i < totalElements; i += blockDim.x) {
        int val = matrix[i];
        if (val <= maxV) {
            atomicAdd(&freqArray[val], 1); // Atomic to avoid race
        }
    }
    __syncthreads();
    
	// prefixSumArray[0] = 0;
    // for (int i = 1; i <= maxV; i++) {
    //     prefixSumArray[i] = prefixSumArray[i - 1] + freqArray[i - 1];
    // }
    if (tid == 0) {
        prefixSumArray[0] = 0;
        for (int i = 1; i <= maxV; i++) {
            prefixSumArray[i] = prefixSumArray[i - 1] + freqArray[i - 1];
        }
    }
    __syncthreads();

	for (int val = 0; val <= maxV; val++) {
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
                //cout<<input[pos-1]<<"|";
            }
        }
        //cout<<endl;
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
    //cout<<maxGlobal<<endl;
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
                //cout<<input[pos-1]<<"|";
            }
        }
        //cout<<endl;
    }

	cudaFree(device_input);
    cudaFree(device_range);
    cudaFree(device_rows);
    cudaFree(device_cols);
    delete[] input;

	return matrices;
}

int main() {
    // Example usage:
    // A 4x3 matrix as given in the problem statement.
    vector<vector<vector<int>>> matrices = {{
        {1, 3, 2},
        {4, 6, 5},
        {7, 9, 8},
        {9, 7, 1}
    }};
    // Upper bound for this matrix's elements is 9.
    vector<int> range = {9};

    vector<vector<vector<int>>> result = modify(matrices, range);

    // Print the modified matrix.
    for (auto& row : result[0]) {
        for (auto val : row)
            cout << val << " ";
        cout << "\n";
    }
    return 0;
}
