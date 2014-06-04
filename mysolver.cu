#include "cuda.h"
#include "cuda_runtime_api.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "globalVars.h"
#include "aux.cu"

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
  float tStart = 0.0, tStop = 35.0;
  float *spkTimes, *vm = NULL, *vstart; // 500 time steps
  int *nSpks, *spkNeuronIds, kNeuron, nSteps, i, k;
  float *dev_vm = NULL, *dev_spkTimes, *dev_vstart;
  int *dev_conVec, *dev_nSpks, *dev_spkNeuronIds;
  FILE *fp;
  float *host_isynap, *dev_isynap;
  int *conVec;
  curandState *devStates;
  cudaEvent_t start0, stop0;
  float elapsedTime;
  cudaError_t devErr;
  /* ================= INITIALIZE ===============================================*/
  nSteps = (tStop - tStart) / DT;
  nSteps = 800;
  printf("\n N = %d NE = %d NI = %d nSteps = %d\n\n", N_NEURONS, NE, NI, nSteps);
  /* ================== SETUP TIMER EVENTS ON DEVICE ==============================*/
  cudaEventCreate(&stop0); cudaEventCreate(&start0);
  cudaEventRecord(start0, 0);
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 128;
  int BlocksPerGrid = (N_NEURONS + ThreadsPerBlock - 1) / ThreadsPerBlock;
  /* ================= ALLOCATE PAGELOCKED MEMORY ON HOST =========================*/
  cudaCheck(cudaMallocHost((void **)&spkTimes, MAX_SPKS  * sizeof(*spkTimes)));
  cudaCheck(cudaMallocHost((void **)&host_isynap, nSteps * N_NEURONS * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&vm, nSteps * N_NEURONS * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&nSpks, sizeof(*nSpks)));
  cudaCheck(cudaMallocHost((void **)&spkNeuronIds, MAX_SPKS * sizeof(spkNeuronIds)));
  cudaCheck(cudaMallocHost((void **)&vstart, N_STATEVARS * N_NEURONS * sizeof(float)));
  cudaCheck(cudaMallocHost((void **)&conVec, N_NEURONS * N_NEURONS * sizeof(int)));
  /* ================= ALLOCATE GLOBAL MEMORY ON DEVICE ===========================*/
  cudaCheck(cudaMalloc((void **)&dev_conVec, N_NEURONS * N_NEURONS * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_vm, nSteps * N_NEURONS * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&dev_isynap, nSteps * N_NEURONS * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&dev_spkTimes, MAX_SPKS * sizeof(*dev_spkTimes)));
  cudaCheck(cudaMalloc((void **)&dev_nSpks, sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_spkNeuronIds, MAX_SPKS * sizeof(*dev_spkNeuronIds)));
  cudaCheck(cudaMalloc((void **)&dev_vstart, N_STATEVARS * N_NEURONS * sizeof(*dev_vstart)));
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  printf("GPU memory allocation successful ! \n ");
  printf("&dev_conVec = %p \n", dev_conVec);
  printf("&devStates = %p\n", devStates);
  printf("&conVec = %p\n", conVec); 
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
  /*===================== GENERATE CONNECTION MATRIX ====================================*/
  cudaCheck(cudaMemset(dev_conVec, 0, N_NEURONS * N_NEURONS * sizeof(int)));
  printf("\n launching rand generator setup kernel\n");
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  printf("\n launching connection matrix geneting kernel with seed %ld ...", time(NULL));
  fflush(stdout);
  kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVec);
  printf(" Done! \n");

  cudaCheck(cudaMemcpy(conVec, dev_conVec, N_NEURONS * N_NEURONS * sizeof(int), cudaMemcpyDeviceToHost));
  /* ==================== INTEGRATE ODEs ON GPU ==========================================*/
    /* invoke device on this block/thread grid */
  rkdumb <<<BlocksPerGrid,ThreadsPerBlock>>> (dev_vstart, N_STATEVARS, tStart, tStop, nSteps, dev_nSpks, dev_spkTimes, dev_spkNeuronIds, dev_vm, dev_conVec, dev_isynap);
    cudaCheckLastError("rkdumb kernel failed");
    printf("kernel done \n");
  // cpy results 
  cudaCheck(cudaMemcpy(nSpks, dev_nSpks, sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(spkTimes, dev_spkTimes, MAX_SPKS * sizeof(float), cudaMemcpyDeviceToHost));
  printf("\n nSpks = %d\n", *nSpks);
  printf("MAX SPKS stored on GPU = %d \n", MAX_SPKS); 
 
  // for(i = 0; i< MAX_SPKS; ++i) { 
  //     printf("spk time = %f\n", spkTimes[i]);
  // }
  /*==================== COPY RESULTS TO HOST =================================================*/
  cudaCheck(cudaMemcpy(spkNeuronIds, dev_spkNeuronIds, MAX_SPKS * sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(vm, dev_vm, nSteps * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(host_isynap, dev_isynap, nSteps * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  /* ================= RECORD COMPUTE TIME ====================================================*/
  cudaEventRecord(stop0, 0);
  cudaEventSynchronize(stop0);
  if((devErr = cudaEventElapsedTime(&elapsedTime, start0, stop0)) == cudaSuccess) {
    printf("elapsed time = %fms \n", elapsedTime);
  }
  cudaCheck(cudaEventDestroy(start0));
  cudaCheck(cudaEventDestroy(stop0));
  /* ================= SAVE TO DISK =============================================================*/
  fp = fopen("vm", "w");
  for(i = 0; i < nSteps; ++i) {
    for(k = 0; k < N_NEURONS; ++k) {
      fprintf(fp, "%f %f ", vm[k + i *  N_NEURONS], host_isynap[k + i * N_NEURONS]);
    }
    fprintf(fp, "\n");
  }
  printf("\n");
  for(i = 0; i < N_NEURONS; ++i) {
    for(k = 0; k < N_NEURONS; ++k) {
      printf("%d ", conVec[i + N_NEURONS *k]);
    }
    printf("\n");
  }
  /*================== CLEANUP ===================================================================*/
  fclose(fp);
  cudaCheck(cudaFreeHost(vm));
  cudaCheck(cudaFreeHost(host_isynap));
  cudaCheck(cudaFreeHost(spkTimes));
  cudaCheck(cudaFreeHost(spkNeuronIds));
  cudaCheck(cudaFreeHost(nSpks));
  cudaCheck(cudaFreeHost(vstart));
  cudaCheck(cudaFree(dev_vm));
  cudaCheck(cudaFree(dev_isynap));
  cudaCheck(cudaFree(dev_spkNeuronIds));
  cudaCheck(cudaFree(dev_spkTimes));
  cudaCheck(cudaFree(dev_conVec));
  cudaCheck(cudaFree(dev_vstart));
  cudaCheck(cudaFree(dev_nSpks));
  cudaCheck(cudaFree(devStates));
  return EXIT_SUCCESS;
}

