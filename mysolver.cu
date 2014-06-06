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

int main(int argc, char *argv[]) {
  float tStart = 0.0, tStop = 100.0;
  float *spkTimes, *vm = NULL;// *vstart; // 500 time steps
  int *nSpks, *spkNeuronIds, nSteps, i, k, lastNStepsToStore;
  float *dev_vm = NULL, *dev_spkTimes;
  int *dev_conVec, *dev_nSpks, *dev_spkNeuronIds;
  FILE *fp, *fpConMat, *fpSpkTimes, *fpElapsedTime;
  float *host_isynap, *dev_isynap;
  int *conVec;
  curandState *devStates;
  cudaEvent_t start0, stop0;
  float elapsedTime;
  /*PARSE INPUTS*/
  //  if(argc >1) {
    //    N_NEURONS = 

  /* ================= INITIALIZE ===============================================*/
  nSteps = (tStop - tStart) / DT;
  lastNStepsToStore = (int)floor(STORE_LAST_T_MILLISEC  / DT);
  //  nSteps = 800;
  printf("\n N  = %d \n NE = %d \n NI = %d \n K  = %d \n nSteps = %d\n\n", N_NEURONS, NE, NI, (int)K, nSteps);
  /* ================== SETUP TIMER EVENTS ON DEVICE ==============================*/
  cudaEventCreate(&stop0); cudaEventCreate(&start0);
  cudaEventRecord(start0, 0);
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 128;
  int BlocksPerGrid = (N_NEURONS + ThreadsPerBlock - 1) / ThreadsPerBlock;
  /* ================= ALLOCATE PAGELOCKED MEMORY ON HOST =========================*/
  cudaCheck(cudaMallocHost((void **)&spkTimes, MAX_SPKS  * sizeof(*spkTimes)));
  cudaCheck(cudaMallocHost((void **)&host_isynap, nSteps * N_NEURONS * sizeof(*host_isynap)));
  cudaCheck(cudaMallocHost((void **)&vm,  lastNStepsToStore * N_NEURONS * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&nSpks, sizeof(*nSpks)));
  cudaCheck(cudaMallocHost((void **)&spkNeuronIds, MAX_SPKS * sizeof(spkNeuronIds)));
  /*cudaCheck(cudaMallocHost((void **)&vstart, N_STATEVARS * N_NEURONS * sizeof(float)));*/
  cudaCheck(cudaMallocHost((void **)&conVec, N_NEURONS * N_NEURONS * sizeof(int)));
  /* ================= ALLOCATE GLOBAL MEMORY ON DEVICE ===========================*/
  cudaCheck(cudaMalloc((void **)&dev_conVec, N_NEURONS * N_NEURONS * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_vm, lastNStepsToStore * N_NEURONS * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&dev_isynap, nSteps * N_NEURONS * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&dev_spkTimes, MAX_SPKS * sizeof(*dev_spkTimes)));
  cudaCheck(cudaMalloc((void **)&dev_nSpks, sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_spkNeuronIds, MAX_SPKS * sizeof(*dev_spkNeuronIds)));
 /*cudaCheck(cudaMalloc((void **)&dev_vstart, N_STATEVARS * N_NEURONS * sizeof(*dev_vstart)));*/
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  printf(" GPU memory allocation successful ! \n ");
  *nSpks = 0;
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
  printf("\n launching Simulation kernel ...");
  fflush(stdout);
  //  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL) + 23);
  rkdumb <<<BlocksPerGrid,ThreadsPerBlock>>> (tStart, tStop, nSteps, dev_nSpks, dev_spkTimes, dev_spkNeuronIds, dev_vm, dev_conVec, dev_isynap, devStates);
  cudaCheckLastError("rkdumb kernel failed");
  /*==================== COPY RESULTS TO HOST =================================================*/
  cudaCheck(cudaMemcpy(nSpks, dev_nSpks, sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(spkTimes, dev_spkTimes, MAX_SPKS * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(spkNeuronIds, dev_spkNeuronIds, MAX_SPKS * sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(vm, dev_vm, lastNStepsToStore * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(host_isynap, dev_isynap, nSteps * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  /* ================= RECORD COMPUTE TIME ====================================================*/
  cudaEventRecord(stop0, 0);
  cudaEventSynchronize(stop0);
  printf(" Done ! \n");
  cudaEventElapsedTime(&elapsedTime, start0, stop0);
  printf("\n elapsed time = %fms \n", elapsedTime);
  fpElapsedTime = fopen("elapsedTime.csv", "a+");
  fprintf(fpElapsedTime, "%f %d\n", elapsedTime, *nSpks);
  cudaCheck(cudaEventDestroy(start0));
  cudaCheck(cudaEventDestroy(stop0));
  printf("\n nSpks = %d\n", *nSpks);
  printf(" MAX SPKS stored on GPU = %d \n", MAX_SPKS); 
  printf("\n Simulation completed ! \n");
  /* ================= SAVE TO DISK =============================================================*/
  printf(" saving results to disk ..."); 
  fflush(stdout);
  fp = fopen("vm.csv", "w");
  for(i = 0; i < lastNStepsToStore; ++i) {
    for(k = 0; k < N_NEURONS; ++k) {
      fprintf(fp, "%f %f ", vm[k + i *  N_NEURONS], host_isynap[k + i * N_NEURONS]);
    }
    fprintf(fp, "\n");
  }
  fpConMat = fopen("conMat.csv", "w");
  for(i = 0; i < N_NEURONS; ++i) {
    for(k = 0; k < N_NEURONS; ++k) {
      fprintf(fpConMat, "%d ", conVec[i + N_NEURONS *k]);
    }
    fprintf(fpConMat, "\n");
  }
  fpSpkTimes = fopen("spkTimes.csv", "w");
  for(i = 1; i < *nSpks; ++i) {
    fprintf(fpSpkTimes, "%f %f\n", spkTimes[i], (float)spkNeuronIds[i] + 1);
  }
  printf("Done!\n");  
  if(*nSpks > MAX_SPKS) {
    printf("\n ***** WARNING MAX_SPKS EXCEEDED limit of %d *****\n", MAX_SPKS);
  }
  /*================== CLEANUP ===================================================================*/
  fclose(fpElapsedTime);
  fclose(fpSpkTimes);
  fclose(fpConMat);
  fclose(fp);
  cudaCheck(cudaFreeHost(vm));
  cudaCheck(cudaFreeHost(host_isynap));
  cudaCheck(cudaFreeHost(spkTimes));
  cudaCheck(cudaFreeHost(spkNeuronIds));
  cudaCheck(cudaFreeHost(nSpks));
  /*  cudaCheck(cudaFreeHost(vstart));*/
  cudaCheck(cudaFree(dev_vm));
  cudaCheck(cudaFree(dev_isynap));
  cudaCheck(cudaFree(dev_spkNeuronIds));
  cudaCheck(cudaFree(dev_spkTimes));
  cudaCheck(cudaFree(dev_conVec));
  /*  cudaCheck(cudaFree(dev_vstart));*/
  cudaCheck(cudaFree(dev_nSpks));
  cudaCheck(cudaFree(devStates));
  return EXIT_SUCCESS;
}

