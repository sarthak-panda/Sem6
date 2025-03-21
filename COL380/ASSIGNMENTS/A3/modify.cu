#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using namespace std;

__global__ void MyKernelFunc(){
	
}

vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& range){
	//let us start with naive approach
	for(int k=0;k<matrices.size();k++){//let us process the matrices sequentially
		vector<vector<int>>mat=matrices[k];
		int maxV=range[k];
		vector<int>freqArray(maxV+1,0);
		vector<int>prefixSumArray(maxV+1,0);
		int row=mat.size();
		int col=mat[0].size();
		for(int i=0;i<row;i++){
			for(int j=0;j<col;j++){
				freqArray[mat[i][j]]++;
			}
		}
		prefixSumArray[0] = freqArray[0];
        for (int i = 1; i <= maxV; i++) {
            prefixSumArray[i] = prefixSumArray[i - 1] + freqArray[i];
        }
		int r=0;
		int c=0;
		for(int i=0;i<maxV+1;i++){
			for(int j=0;j<freqArray[i];j++){
				mat[r][c]=i;
				if(r<row){
					r++;
				}else{
					r=0;
					c++;
				}
			}
		}
		matrices[k]=mat;
	}
	return matrices;
}