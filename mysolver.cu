#include "cuda.h"
#include "cuda_runtime_api.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "globalVars.h"
#include "aux.cu"

//#define N_NEURONS 1

void __cudaCheck(cudaError err, const char* file, const int line);
#define cudaCheck(err) __cudaCheck (err, __FILE__, __LINE__)

void __cudaCheckLastError(const char* errorMessage, const char* file, const int line);
#define cudaCheckLastError(msg) __cudaCheckLastError (msg, __FILE__, __LINE__)

void __cudaCheck(cudaError err, const char *file, const int line)
{
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
      file, line, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

void __cudaCheckLastError(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
      file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}
 

int main() {
  float tStart = 0.0, tStop = 100.0;
  float *spkTimes, *vm = NULL, *vstart; // 500 time steps
  int *nSpks, *spkNeuronIds, kNeuron, nSteps, i, k;
  float *dev_vm = NULL, *dev_spkTimes, *dev_vstart;
  int *dev_conVec, *dev_nSpks, *dev_spkNeuronIds;
  FILE *fp;
  //  float *dev_gE, *dev_gI, *gE, *gI;
  //  cudaEvent_t start, stop;
  //  float elapsedTime;
  float *host_isynap, *dev_isynap;
  
  
  int conVec[] = {0, 1, 0, 0};
  //  curandState *devStates;
  // ================= INITIALIZE ===============================================\\

  nSteps = (tStop - tStart) / DT;

  printf("\n N = %d NE = %d NI = %d nSteps = %d\n\n", N_NEURONS, NE, NI, nSteps);

  //================== SETUP TIMER EVENTS ON DEVICE ==============================\\
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);
  // =================== GENERATE CONNECTION MATRIX ===============================\\
  //  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));    
  cudaCheck(cudaMalloc((void **)&dev_conVec, N_NEURONS * N_NEURONS * sizeof(int)));
  //  cudaCheck(cudaMallocHost((void **)&conVec, N_NEURONS * N_NEURONS * sizeof(int)));
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 128;
  int BlocksPerGrid = (N_NEURONS + ThreadsPerBlock - 1) / ThreadsPerBlock;

  // printf("\n launchint kernel with %d ThreadsPerBlock & %d BlocksPerGrid", ThreadsPerBlock, BlocksPerGrid);
  // // call CUDA : setup random number generator
  // setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));  
  // // call CUDA : generate matrix
  // kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, nNeurons, dev_conVec);


  cudaCheck(cudaMemcpy(dev_conVec, conVec, N_NEURONS * N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
//  cudaCheck(cudaMemcpy(conVec, dev_conVec, N_NEURONS * N_NEURONS * sizeof(int), cudaMemcpyDeviceToHost));


  //  cudaCheck(cudaFree(devStates)); // do not free if more random numbers are needed per thread 
  // ================= ALLOCATE PAGELOCKED MEMORY ON HOST =========================\\
  //  cudaCheck(cudaMallocHost((void **)&gE, N_NEURONS * sizeof(*dev_gE)));
  //  cudaCheck(cudaMallocHost((void **)&gI, N_NEURONS  * sizeof(*dev_gI)));

  cudaCheck(cudaMallocHost((void **)&host_isynap, nSteps * N_NEURONS * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&spkTimes, MAX_SPKS  * sizeof(*spkTimes)));
  cudaCheck(cudaMallocHost((void **)&vm, nSteps * N_NEURONS * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&nSpks, sizeof(*nSpks)));
  cudaCheck(cudaMallocHost((void **)&spkNeuronIds, MAX_SPKS * sizeof(spkNeuronIds)));
  cudaCheck(cudaMallocHost((void **)&vstart, N_STATEVARS * N_NEURONS * sizeof(float)));
  // ================= ALLOCATE GLOBAL MEMORY ON DEVICE ===========================\\
  //  cudaCheck(cudaMalloc((void **)&dev_gE, N_NEURONS * sizeof(*dev_gE)));
  //  cudaCheck(cudaMalloc((void **)&dev_gI, N_NEURONS  * sizeof(*dev_gI)));
  cudaCheck(cudaMalloc((void **)&dev_isynap, nSteps * N_NEURONS * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&dev_vm, nSteps * N_NEURONS * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&dev_spkTimes, MAX_SPKS * sizeof(*dev_spkTimes)));
  cudaCheck(cudaMalloc((void **)&dev_nSpks, sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_spkNeuronIds, MAX_SPKS * sizeof(*dev_spkNeuronIds)));
  cudaCheck(cudaMalloc((void **)&dev_vstart, N_STATEVARS * N_NEURONS * sizeof(*dev_vstart)));
  printf("GPU memory allocation successful ! \n ");
  printf("&dev_vm = %p \n", dev_vm);

  for(kNeuron = 0; kNeuron < N_Neurons; ++kNeuron) {
    int clmNo =  kNeuron * N_STATEVARS;
    vstart[0 + clmNo] = -50; //-70 +  40 * CudaURand(); // Vm(0) ~ U(-70, -30)
    vstart[1 + clmNo] = 0.3176;
    vstart[2 + clmNo] = 0.1;
    vstart[3 + clmNo] = 0.5961;
  }
  *nSpks = 0;
  cudaCheck(cudaMemcpy(dev_vstart, vstart, N_STATEVARS * N_Neurons * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_nSpks, nSpks, sizeof(int), cudaMemcpyHostToDevice));
  // ==================== INTEGRATE ODEs ON GPU ========================================== \\
    /* invoke device on this block/thread grid */
  rkdumb <<<BlocksPerGrid,ThreadsPerBlock>>> (dev_vstart, N_STATEVARS, tStart, tStop, nSteps, dev_nSpks, dev_spkTimes, dev_spkNeuronIds, dev_vm, dev_conVec, dev_isynap);
    cudaCheckLastError("rkdumb kernel failed");
    printf("kernel done \n");
  // cpy results 
  cudaCheck(cudaMemcpy(nSpks, dev_nSpks, sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(spkTimes, dev_spkTimes, MAX_SPKS * sizeof(float), cudaMemcpyDeviceToHost));
  printf("\n nSpks = %d\n", *nSpks);
  printf("MAX SPKS stored on GPU = %d \n", MAX_SPKS); 
  for(i = 0; i< MAX_SPKS; ++i) { 
      printf("spk time = %f\n", spkTimes[i]);
  }
  cudaCheck(cudaMemcpy(spkNeuronIds, dev_spkNeuronIds, MAX_SPKS * sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(vm, dev_vm, nSteps * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(host_isynap, dev_isynap, nSteps * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  //  cudaCheck(cudaMemcpy(gI, dev_gI, N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  //  cudaCheck(cudaMemcpy(gE, dev_gE, N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&elapsedTime, start, stop);
  // printf("elapsed time = %fms \n", elapsedTime);
  // cudaCheck(cudaEventDestroy(start));
  // cudaCheck(cudaEventDestroy(stop));

  fp = fopen("vm", "w");
  for(i = 0; i < nSteps; ++i) {
    for(k = 0; k < N_NEURONS; ++k) {
      fprintf(fp, "%f %f ", vm[k + i *  N_NEURONS], host_isynap[k + i * N_NEURONS]);
    }
    fprintf(fp, "\n");
  }
  
  fclose(fp);
  cudaCheck(cudaFreeHost(vm));
  cudaCheck(cudaFreeHost(spkTimes));
  cudaCheck(cudaFreeHost(spkNeuronIds));
  cudaCheck(cudaFreeHost(nSpks));
  cudaCheck(cudaFreeHost(vstart));
  cudaCheck(cudaFree(dev_vm));
  cudaCheck(cudaFree(dev_spkNeuronIds));
  cudaCheck(cudaFree(dev_spkTimes));
  cudaCheck(cudaFree(dev_conVec));
   cudaCheck(cudaFree(dev_vstart));
   //  free(vstart);
  cudaCheck(cudaFree(dev_nSpks));
  return EXIT_SUCCESS;
}

