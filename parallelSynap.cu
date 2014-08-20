#include "cuda.h"
#include "globalVars.h"


__global__ void parallelSynUpdate(int *dev_spkCountMatrix, int localSpkId, int localSparseIdx) {
  unsigned int mNeuron = threadIdx.x + blockIdx.x * blockDim.x;
  dev_spkCountMatrix[dev_sparseConVec[localSparseIdx + mNeuron] + N_NEURONS + localSpkId] += 1;
}


__global__ void computeG_Optimal01() {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  int localSpkId, localSparseIdx, nThreads = 256;
  if(mNeuron < N_NEURONS) {
    if(dev_IF_SPK[mNeuron]) {  
      localSpkId = dev_prevStepSpkIdx[mNeuron];
      localSparseIdx = dev_sparseIdx[mNeuron];
      if(mNeuron < NE) {       
	 parallelSynUpdate<<<((int)K + nThreads) / nThreads, nThreads>>>(&dev_ESpkCountMat[0], localSpkId, localSparseIdx);
      }
      else {
	parallelSynUpdate<<<((int)K + nThreads) / nThreads, nThreads>>>(&dev_ISpkCountMat[0], localSpkId, localSparseIdx);
      }
    }
  }
} 


__global__ void ChildKernel(void* data){
  //Operate on data
}
__global__ void ParentKernel(void *data){
  ChildKernel<<<16, 1>>>(data);
}
// In Host Code
/*ParentKernel<<<256, 64>>(data);*/
