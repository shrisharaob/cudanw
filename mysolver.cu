/* cuda network simulation 
   History :                    
    created: Shrisha
   Makefile included for build on CC=3.5
*/
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
  double tStart = 0.0, tStop = 100.0;
  double *spkTimes, *vm = NULL, host_theta = 0.0; /* *vstart; 500 time steps */
  int *nSpks, *spkNeuronIds, nSteps, i, k, lastNStepsToStore;
  double *dev_vm = NULL, *dev_spkTimes, *dev_time = NULL, *host_time;
  int *dev_conVec, *dev_nSpks, *dev_spkNeuronIds;
  FILE *fp, *fpConMat, *fpSpkTimes, *fpElapsedTime;
  double *host_isynap, *synapticCurrent;
  int *conVec;
  curandState *devStates, *devNormRandState;
  cudaEvent_t start0, stop0;
  float elapsedTime;
  int *dev_sparseVec = NULL, *sparseConVec = NULL;
  int idxVec[N_NEURONS], nPostNeurons[N_NEURONS], *dev_idxVec = NULL, *dev_nPostneuronsPtr = NULL;
  int deviceId = 0;
  devPtr_t devPtrs;
  kernelParams_t kernelParams;
  int IF_SAVE = 1;
  /*PARSE INPUTS*/
  if(argc >1) {
    deviceId = atoi(argv[1]);
    if(argc > 2) {
      IF_SAVE = atoi(argv[2]);
    }
    if(argc > 3) {
      host_theta = atof(argv[3]);
    }
  }
  printf("\n Computing on GPU%d \n", deviceId);
  cudaCheck(cudaSetDevice(deviceId));
  cudaMemcpyToSymbol(theta, &host_theta, sizeof(host_theta));
  /* ================= INITIALIZE ===============================================*/
  nSteps = (tStop - tStart) / DT;
  lastNStepsToStore = (int)floor(STORE_LAST_T_MILLISEC  / DT);
  //  nSteps = 800;
  printf("\n N  = %d \n NE = %d \n NI = %d \n K  = %d \n tStop = %3.2f seconds nSteps = %d\n\n", N_NEURONS, NE, NI, (int)K, tStop / 1000.0, nSteps);
  printf(" theta = %2.1f\n", host_theta);
  
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 128;
  int BlocksPerGrid = (N_NEURONS + ThreadsPerBlock - 1) / ThreadsPerBlock;
  printf("Threads per block : %d, Blocks per grid : %d \n", ThreadsPerBlock, BlocksPerGrid);
  /*INITIALIZE RND GENERATORS FOR ibf & iff */
  setupBGCurGenerator<<<BlocksPerGrid, ThreadsPerBlock>>>(time(NULL));
  setupIFFRndGenerator<<<BlocksPerGrid, ThreadsPerBlock>>>(time(NULL));
  /*Generate frozen FF input approximat*/
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  cudaCheck(cudaMalloc((void **)&devNormRandState, N_NEURONS * sizeof(curandState)));
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devNormRandState, time(NULL));
  AuxRffTotal<<<BlocksPerGrid, ThreadsPerBlock>>>(devNormRandState, devStates);
  cudaCheck(cudaFree(devNormRandState));
  /* gENERATE CONNECTION MATRIX */
  cudaCheck(cudaMalloc((void **)&dev_conVec, N_NEURONS * N_NEURONS * sizeof(int)));
  cudaCheck(cudaMallocHost((void **)&conVec, N_NEURONS * N_NEURONS * sizeof(int)));  
  cudaCheck(cudaMemset(dev_conVec, 0, N_NEURONS * N_NEURONS * sizeof(int)));
  printf("\n launching rand generator setup kernel\n");
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  printf("\n launching connection matrix geneting kernel with seed %ld ...", time(NULL));
  fflush(stdout);
  kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVec);
  printf(" Done! \n");
  cudaCheck(cudaMemcpy(conVec, dev_conVec, N_NEURONS * N_NEURONS * sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaFree(dev_conVec));
  /* SPARSIFY */
  conVec[0] = 0; conVec[1] = 1; conVec[2] = 1;conVec[3] = 0;
  /*conVec[4] = 0;conVec[5] = 1;conVec[6] = 1;conVec[7] = 1;
  conVec[8] = 1;*/ /*conVec[9] = 0;*/
  /*conVec[10] = 0;conVec[11] = 1;
    conVec[12]= 0;conVec[13] = 0;conVec[14] = 0;conVec[15] = 0;*/
  cudaCheck(cudaGetSymbolAddress((void **)&dev_sparseVec, dev_sparseConVec));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_idxVec, dev_sparseIdx));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nPostneuronsPtr, dev_nPostNeurons));
  cudaCheck(cudaMallocHost((void **)&sparseConVec, N_NEURONS * (2 * K + 1) * sizeof(int)));  
  /*  cudaCheck(cudaMalloc((void **)&dev_sparseVec, N_NEURONS * ((int)2 * K + 1)* sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_idxVec, N_NEURONS * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_nPostneuronsPtr, N_NEURONS * sizeof(int)));*/
  GenSparseMat(conVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);
  cudaCheck(cudaMemcpy(dev_sparseVec, sparseConVec, N_NEURONS * (2 * K + 1) * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_idxVec, idxVec, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_nPostneuronsPtr, nPostNeurons, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
 /* ================= ALLOCATE PAGELOCKED MEMORY ON HOST =========================*/
  cudaCheck(cudaMallocHost((void **)&spkTimes, MAX_SPKS  * sizeof(*spkTimes)));
  cudaCheck(cudaMallocHost((void **)&host_isynap, lastNStepsToStore * N_NEURONS * sizeof(*host_isynap)));
  cudaCheck(cudaMallocHost((void **)&vm,  lastNStepsToStore * N_NEURONS * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&host_time,  lastNStepsToStore * N_NEURONS * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&nSpks, sizeof(*nSpks)));
  cudaCheck(cudaMallocHost((void **)&spkNeuronIds, MAX_SPKS * sizeof(*spkNeuronIds)));
  /* ================= ALLOCATE GLOBAL MEMORY ON DEVICE ===========================*/
  /*cudaCheck(cudaMalloc((void **)&dev_conVec, N_NEURONS * N_NEURONS * sizeof(int)));*/
  cudaCheck(cudaMalloc((void **)&dev_vm, lastNStepsToStore * N_NEURONS * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&dev_time, lastNStepsToStore * N_NEURONS * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&synapticCurrent, lastNStepsToStore * N_NEURONS * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&dev_spkTimes, MAX_SPKS * sizeof(*dev_spkTimes)));
  cudaCheck(cudaMalloc((void **)&dev_nSpks, sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_spkNeuronIds, MAX_SPKS * sizeof(*dev_spkNeuronIds)));
  cudaCheck(cudaMemset(dev_spkTimes, 0, MAX_SPKS * sizeof(*dev_spkTimes)));
  cudaCheck(cudaMemset(dev_spkNeuronIds, 0.0f, MAX_SPKS * sizeof(*dev_spkNeuronIds)));
  printf(" GPU memory allocation successful ! \n ");
  devPtrs.dev_conVec = dev_conVec;
  devPtrs.dev_spkNeuronIds = dev_spkNeuronIds;
  devPtrs.dev_vm = dev_vm;
  devPtrs.dev_nSpks = dev_nSpks;
  devPtrs.dev_spkNeuronIds = dev_spkNeuronIds;
  devPtrs.dev_spkTimes = dev_spkTimes;
  devPtrs.synapticCurrent = synapticCurrent;
  devPtrs.devStates = devStates;
  /*  devPtrs.dev_sparseConVec = dev_sparseVec;
  devPtrs.dev_nPostNeurons = dev_nPostneuronsPtr;
  devPtrs.dev_sparseIdx = dev_idxVec;*/
  devPtrs.dev_time = dev_time;
  *nSpks = 0;
  cudaCheck(cudaMemcpy(dev_nSpks, nSpks, sizeof(int), cudaMemcpyHostToDevice));
  printf("devspk ptrs: %p %p \n", dev_spkTimes, dev_spkNeuronIds);
  /* INTEGRATE ODEs ON GPU*/
  /* invoke device on this block/thread grid */
  kernelParams.nSteps = nSteps;
  kernelParams.tStop = tStop;
  kernelParams.tStart = tStart;
  printf("\n launching Simulation kernel ...");
  fflush(stdout);
  /* SETUP TIMER EVENTS ON DEVICE */
  cudaEventCreate(&stop0); cudaEventCreate(&start0);
  cudaEventRecord(start0, 0);
  /* TIME LOOP */
  for(k = 0; k < nSteps; ++k) { 
    devPtrs.k = k;
    rkdumbPretty<<<BlocksPerGrid, ThreadsPerBlock>>> (kernelParams, devPtrs);
    expDecay<<<BlocksPerGrid, ThreadsPerBlock>>>();
    computeConductance<<<BlocksPerGrid, ThreadsPerBlock>>>();
    computeIsynap<<<BlocksPerGrid, ThreadsPerBlock>>>();
  }
  /* RECORD COMPUTE TIME */
  cudaCheckLastError("rkdumb kernel failed");
  cudaEventRecord(stop0, 0);
  cudaEventSynchronize(stop0);
  printf(" Done ! \n");
  cudaEventElapsedTime(&elapsedTime, start0, stop0);
  printf("\n elapsed time = %fms \n", elapsedTime);
  cudaCheck(cudaEventDestroy(start0));
  cudaCheck(cudaEventDestroy(stop0));
  /*==================== COPY RESULTS TO HOST =================================================*/
  cudaCheck(cudaMemcpy(nSpks, dev_nSpks, sizeof(int), cudaMemcpyDeviceToHost));
  printf("devspk ptrs: %p %p \n", dev_spkTimes, dev_spkNeuronIds);
  cudaCheck(cudaMemcpy(spkTimes, dev_spkTimes, MAX_SPKS * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(spkNeuronIds, dev_spkNeuronIds, MAX_SPKS * sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(vm, dev_vm, lastNStepsToStore * N_NEURONS * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(host_time, dev_time, lastNStepsToStore * N_NEURONS * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(host_isynap, synapticCurrent, lastNStepsToStore * N_NEURONS * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(vm, dev_vm, lastNStepsToStore * N_NEURONS * sizeof(double), cudaMemcpyDeviceToHost));
  double curE[5 * 4000], curI[5 * 4000], ibgCur[5 * 4000], *dev_curE, *dev_curI, *dev_ibg, curIff[5000], *dev_curiff;
  cudaCheck(cudaGetSymbolAddress((void **)&dev_curE, glbCurE));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_curI, glbCurI));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_ibg, dev_bgCur));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_curiff, dev_iff));
  cudaCheck(cudaMemcpy(curE, dev_curE, 5 * 4000 * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(curI, dev_curI, 5 * 4000 * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(ibgCur, dev_ibg, 5 * 4000 * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(curIff, dev_curiff, 5000 * sizeof(double), cudaMemcpyDeviceToHost));
  printf("\n nSpks = %d\n", *nSpks);
  printf(" MAX SPKS stored on GPU = %d \n", MAX_SPKS); 
  printf("\n Simulation completed ! \n");
  fpElapsedTime = fopen("elapsedTime.csv", "a+");
  fprintf(fpElapsedTime, "%d %f %d\n", N_NEURONS, elapsedTime, *nSpks);
  fclose(fpElapsedTime);
  /* ================= SAVE TO DISK =============================================================*/
  if(IF_SAVE) {  
    printf(" saving results to disk ... "); 
    fflush(stdout);
    fpSpkTimes = fopen("spkTimes.csv", "w");
    int totalNSpks = *nSpks;
    printf(" saving spikes ...");
    fflush(stdout);
    if(*nSpks > MAX_SPKS) {
      totalNSpks = MAX_SPKS;
      printf("\n ***** WARNING MAX_SPKS EXCEEDED limit of %d *****\n", MAX_SPKS);
    }
    for(i = 1; i <= totalNSpks; ++i) {
      fprintf(fpSpkTimes, "%f;%f\n", spkTimes[i], (double)spkNeuronIds[i]);
    }
    printf("Done!\n");
    fclose(fpSpkTimes);

    fp = fopen("vm.csv", "w");
    for(i = 0; i < lastNStepsToStore; ++i) {
      fprintf(fp, "%f ", host_time[i]);
      for(k = 0; k < N_NEURONS; ++k) {
	/*	fprintf(fp, "%f %f ", vm[k + i *  N_NEURONS], host_isynap[k + i * N_NEURONS]);*/
	fprintf(fp, "%f ", vm[k + i *  N_NEURONS]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
    FILE* fpCur = fopen("currents.csv", "w");
    for(i = 0; i < 5000; ++i) {
      fprintf(fpCur, "%f;%f;%f;%f\n", curE[i], curI[i], ibgCur[i], curIff[i]);
    }
    fclose(fpCur);
    /*fpConMat = fopen("conMat.csv", "w");*/
    fpConMat = fopen("conVec.csv", "w");
    fwrite(conVec, size(int), N_NEURONS * N_NEURONS, fpConMat);
    /*for(i = 0; i < N_NEURONS; ++i) {
      for(k = 0; k < N_NEURONS; ++k) {
	fprintf(fpConMat, "%d", conVec[i *  N_NEURONS + k]);
      }
            fprintf(fpConMat, "\n");
      }*/
    fclose(fpConMat);
  }
  /*================== CLEANUP ===================================================================*/
  cudaCheck(cudaFreeHost(vm));
  cudaCheck(cudaFreeHost(host_time));
  cudaCheck(cudaFreeHost(host_isynap));
  cudaCheck(cudaFreeHost(spkTimes));
  cudaCheck(cudaFreeHost(spkNeuronIds));
  cudaCheck(cudaFreeHost(nSpks));
  cudaCheck(cudaFree(dev_vm));
  cudaCheck(cudaFree(dev_time));
  cudaCheck(cudaFree(synapticCurrent));
  cudaCheck(cudaFree(dev_spkNeuronIds));
  cudaCheck(cudaFree(dev_spkTimes));
  cudaCheck(cudaFree(dev_nSpks));
  cudaCheck(cudaFree(devStates));
  /*  cudaCheck(cudaFree(dev_sparseVec));
  cudaCheck(cudaFree(dev_idxVec));
  cudaCheck(cudaFree(dev_nPostneuronsPtr));*/
  return EXIT_SUCCESS;
}

