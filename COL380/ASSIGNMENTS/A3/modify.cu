#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using namespace std;

__global__ void MyKernelFunc(vector<vector<vector<int>>>& matrices, vector<int>& range){
	int k=threadIdx.x;
	vector<vector<int>>mat=matrices[k];
	int maxV=range[k];
	vector<int>freqArray(maxV+1,0);
	vector<int>prefixSumArray(maxV+1,0);
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			freqArray[mat[i][j]]++;
		}
	}
	/*updated prefixSumArray Method so if at index i ,
	freqArray[i]!=0 then the output matrix starts the value i 
	from row major index prefixSumArray[i] and continues till 
	it hits another non zero frequency array element*/
	prefixSumArray[0] = 0;
	for (int i = 1; i <= maxV; i++) {
		prefixSumArray[i] = prefixSumArray[i - 1] + freqArray[i-1];
	}
	//to use prefixSumArray and freqArray efficiently using CUDA programming
	//write the updated matxrix to mat
	matrices[k]=mat;
	return;
}

vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& range){
	/*let us prefetceh the matrices,range to GPU to lower the bottle-neck of GPU-CPU conversatiomn
	i think i should use i either use cudaMemPrefetchAsync(left, size*sizeof(float), device, NULL);[just wrote signature] 
	or cudaMemcpyAsync(dev_pointer, host_pointer, size, cudaMemcpyHostToDevice, stream);[just wrote signature]
	or something else that fits well, i may need to update the way variable are passes to kernel depending on what i do*/
	int numBlocks=1;
	int numThreads=1;//i think i should len(matrices) because i am thinking threadIdx.x will how come take on all possible values to process entire matrices array in parallel
	MyKernelFunc<<<numBlocks,numThreads>>>(matrices,range);
	cudaDeviceSynchronize();
	return matrices;
}