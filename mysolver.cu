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
#include "cuda_histogram.h"

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
  double tStart = 0.0, tStop = 25000.0;
  double *spkTimes, *vm = NULL, host_theta = 0.0; /* *vstart; 500 time steps */
  int *nSpks, *spkNeuronIds, nSteps, i, k, lastNStepsToStore;
  double *dev_vm = NULL, *dev_spkTimes, *dev_time = NULL, *host_time;
  int *dev_conVec = NULL, *dev_nSpks, *dev_spkNeuronIds;
  FILE *fp, *fpConMat, *fpSpkTimes, *fpElapsedTime;
  double *host_isynap, *synapticCurrent;
  /*  int *conVec;*/
  curandState *devStates, *devNormRandState;
  cudaEvent_t start0, stop0;
  float elapsedTime;
  int *dev_sparseVec = NULL, *sparseConVec = NULL;
  int idxVec[N_NEURONS], nPostNeurons[N_NEURONS], *dev_idxVec = NULL, *dev_nPostneuronsPtr = NULL;
  int deviceId = 0;
  devPtr_t devPtrs;
  kernelParams_t kernelParams;
  int IF_SAVE = 1;
  cudaStream_t stream1;
  cudaCheck(cudaStreamCreate(&stream1));

  /*PARSE INPUTS*/
  if(argc > 1) {
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
  host_theta = PI * host_theta / (180.0); /* convert to radians */
  cudaMemcpyToSymbol(theta, &host_theta, sizeof(host_theta));
  /* ================= INITIALIZE ===============================================*/
  nSteps = (tStop - tStart) / DT;
  lastNStepsToStore = (int)floor(STORE_LAST_T_MILLISEC  / DT);
  //  nSteps = 800;
  printf("\n N  = %llu \n NE = %llu \n NI = %llu \n K  = %d \n tStop = %3.2f seconds nSteps = %d\n\n", N_NEURONS, NE, NI, (int)K, tStop / 1000.0, nSteps);
  
  printf(" theta = %2.3f \n contrast = %2.1f\n ksi = %f\n dt = %f\n", host_theta, HOST_CONTRAST, ETA_E, DT);
  
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
  /*  cudaCheck(cudaMalloc((void **)&dev_conVec, N_NEURONS * N_NEURONS * sizeof(int)));*/
  /*  cudaCheck(cudaMallocHost((void **)&conVec, N_NEURONS * N_NEURONS * sizeof(int)));  */
  /*  cudaCheck(cudaMemset(dev_conVec, 0, N_NEURONS * N_NEURONS * sizeof(int)));
  printf("\n launching rand generator setup kernel\n");
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  printf("\n launching connection matrix geneting kernel with seed %ld ...", time(NULL));
  fflush(stdout);
  kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVec);
  printf(" Done! \n");
  cudaCheck(cudaMemcpy(conVec, dev_conVec, N_NEURONS * N_NEURONS * sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaFree(dev_conVec));*/
  /*  printf("reading convec.dat..."); fflush(stdout);
  FILE *fpConVecFile = fopen("conVec.dat", "rb");
  fread(conVec, sizeof(*conVec), N_NEURONS * N_NEURONS, fpConVecFile);
  fclose(fpConVecFile);
  printf("done ...\n");*/
  /* SPARSIFY */
  /*  conVec[0] = 0; conVec[1] = 0; conVec[2] = 1;conVec[3] = 0;*/
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
  /*  GenSparseMat(conVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);*/
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  fpSparseConVec = fopen("sparseConVec.dat", "rb");
  fpIdxVec = fopen("idxVec.dat", "rb");
  fpNpostNeurons = fopen("nPostNeurons.dat", "rb");
  int dummy;
  dummy = fread(sparseConVec, sizeof(*sparseConVec), N_NEURONS * (2 * (int)K + 1), fpSparseConVec);
  if(dummy != N_NEURONS * (2 * (int)K + 1)) {
    printf("sparseConvec read error ? \n");
  }
  dummy = fread(idxVec, sizeof(*idxVec), N_NEURONS, fpIdxVec);
  dummy = fread(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpSparseConVec);
  fclose(fpIdxVec);
  fclose(fpNpostNeurons);
  /*
    for(i = 0; i < N_NEURONS; ++i) {
      printf("neuron %d projects to : ", i);
      for(int j = 0; j < nPostNeurons[i]; ++j) {
	printf("%d ", sparseConVec[idxVec[i] + j]);
      }
      printf("\n");
    }
  */

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

  /*  printf("devspk ptrs: %p %p \n", dev_spkTimes, dev_spkNeuronIds);*/
  /*===================== GENERATE CONNECTION MATRIX ====================================*/
  /*cudaCheck(cudaMemset(dev_conVec, 0, N_NEURONS * N_NEURONS * sizeof(int)));
  printf("\n launching rand generator setup kernel\n");
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  printf("\n launching connection matrix geneting kernel with seed %ld ...", time(NULL));
  fflush(stdout);
  kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVec);
  printf(" Done! \n");
  cudaCheck(cudaMemcpy(conVec, dev_conVec, N_NEURONS * N_NEURONS * sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheck(cudaFree(dev_conVec));
  GenSparseMat(conVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);
  cudaCheck(cudaMemcpy(dev_sparseVec, sparseConVec, N_NEURONS * (2 * K + 1) * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_idxVec, idxVec, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_nPostneuronsPtr, nPostNeurons, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));*/
  /* ==================== INTEGRATE ODEs ON GPU ==========================================*/
    /* invoke device on this block/thread grid */
  kernelParams.nSteps = nSteps;
  kernelParams.tStop = tStop;
  kernelParams.tStart = tStart;
  printf("\n launching Simulation kernel ...");
  fflush(stdout);
  
  
  
  int *dev_IF_SPK_Ptr = NULL, *dev_prevStepSpkIdxPtr = NULL, *host_IF_SPK = NULL, *host_prevStepSpkIdx = NULL,  *dev_nEPtr = NULL, *dev_nIPtr = NULL;
  int nSpksInPrevStep;
  cudaCheck(cudaMallocHost((void **)&host_IF_SPK, N_NEURONS * sizeof(int)));
  cudaCheck(cudaMallocHost((void **)&host_prevStepSpkIdx, N_NEURONS * sizeof(int)));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_IF_SPK_Ptr, dev_IF_SPK));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_prevStepSpkIdxPtr, dev_prevStepSpkIdx));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nEPtr, dev_ESpkCountMat));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nIPtr, dev_ISpkCountMat));
  for(i = 0; i < N_NEURONS; ++i) {
    host_IF_SPK[i] = 0;
  }
  /* TIME LOOP */
  size_t sizeOfInt = sizeof(int);
  /* SETUP TIMER EVENTS ON DEVICE */
  cudaEventCreate(&stop0); cudaEventCreate(&start0);
  cudaEventRecord(start0, 0);
  unsigned int spksE = 0, spksI = 0;
  FILE *fpIFR = fopen("instant_fr.csv", "w");
  int *histVec = NULL, *dev_histVec = NULL; /* for storing the post-synaptic neurons to be updated */
  int histVecIndx = 0;
  unsigned int histVecLength = N_NEURONS * (int)K;
  if((unsigned long long)K >= NE | (unsigned long long)K >= NI) {
    histVecLength = (unsigned int)(N_NEURONS * N_NEURONS);
  }
  cudaCheck(cudaMallocHost((void **)&histVec, histVecLength * sizeof(*histVec)));
  cudaCheck(cudaMalloc((void **)&dev_histVec, histVecLength * sizeof(*dev_histVec)));
  test_xform xform; // defined in cuda_histogram.h
  test_sumfun sum;  // defined in cuda_histogram.h
  int *dev_histCountE = NULL, *histCountE = NULL, *dev_histCountI = NULL, *histCountI = NULL;;
  cudaCheck(cudaMalloc((void **)&dev_histCountE, sizeof(int) * N_NEURONS));
  cudaCheck(cudaMallocHost((void **)&histCountE, sizeof(int) * N_NEURONS));
  cudaCheck(cudaMalloc((void **)&dev_histCountI, sizeof(int) * N_NEURONS));
  cudaCheck(cudaMallocHost((void **)&histCountI, sizeof(int) * N_NEURONS));
  int tmp;
  printf("\n\n\n\n %d\n\n\n\n", sparseConVec[835584ULL]);
  for(k = 0; k < nSteps; ++k) { 
    /*    cudaCheck(cudaMemsetAsync(dev_nEPtr, 0, N_NEURONS * N_SPKS_IN_PREV_STEP * sizeOfInt, stream1));
	  cudaCheck(cudaMemsetAsync(dev_nIPtr, 0, N_NEURONS * N_SPKS_IN_PREV_STEP * sizeOfInt, stream1));*/
    /*    nSpksInPrevStep = 0;*/
    devPtrs.k = k;
    nSpksInPrevStep = 0;
    histVecIndx = 0;
    for(i = 0; i < N_NEURONS; ++i) {
      histCountI[i] = 0;
      histCountE[i] = 0;
    }

    rkdumbPretty<<<BlocksPerGrid, ThreadsPerBlock>>> (kernelParams, devPtrs);
    cudaCheckLastError("rk");
    if(k > 0) {
      /*      cudaCheck(cudaMemcpy(host_IF_SPK, dev_IF_SPK_Ptr, N_NEURONS * sizeOfInt, cudaMemcpyDeviceToHost));*/
      cudaCheck(cudaMemcpyAsync(host_IF_SPK, dev_IF_SPK_Ptr, N_NEURONS * sizeOfInt, cudaMemcpyDeviceToHost, stream1));
    }
    cudaCheck(cudaStreamSynchronize(stream1));
    /*instantaneous firing rate, rect non-overlapping window */
    for(i = 0; i < N_NEURONS; ++i) {
      if(host_IF_SPK[i]) {
	if(i < NE) {
	  spksE += 1;
	}
	else{
	  spksI += 1;
	}
	/*	    host_prevStepSpkIdx[i] = nSpksInPrevStep;
		    nSpksInPrevStep += 1;*/
      }
    }
    if(!(k%2000)) {
      fprintf(fpIFR, "%f %f \n", ((double)spksE) / (0.05 * (double)NE), ((double)spksI) / (0.05 * (double)NI));fflush(fpIFR);
      fprintf(stdout, "%f %f \n", ((double)spksE) / (0.05 * (double)NE), ((double)spksI) / (0.05 * (double)NI));
      spksE = 0; 
      spksI = 0;
    }
    /*-----------------------------------------------------------------------*/
    expDecay<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_histCountE, dev_histCountI);
    cudaCheckLastError("exp");
    for(i = 0; i < NE; ++i) {
      if(host_IF_SPK[i]){
      nSpksInPrevStep += 1;
        for(int jj = 0; jj < nPostNeurons[i]; ++jj) {
          tmp = sparseConVec[idxVec[i] + jj];
          histVec[histVecIndx++] = tmp;
            /*          histVec[histVecIndx++] = sparseConVec[idxVec[i] + jj];*/
        }
      }
    }
    if(nSpksInPrevStep) {
      cudaCheck(cudaMemcpy(dev_histVec, histVec, histVecIndx * sizeof(int), cudaMemcpyHostToDevice));
      callHistogramKernel<histogram_atomic_inc, 1>(dev_histVec, xform, sum, 0, histVecIndx, 0, &histCountE[0], (int)N_NEURONS);
      /*      cudaCheck(cudaMemcpy(dev_histCountE, histCountE, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));*/
      cudaCheck(cudaMemcpyAsync(dev_histCountE, histCountE, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice, stream1));
    }
    
    histVecIndx = 0;
    nSpksInPrevStep = 0; 
    for(i = NE; i < N_NEURONS; ++i) {
      if(host_IF_SPK[i]){
        nSpksInPrevStep += 1;
        for(int jj = 0; jj < nPostNeurons[i]; ++jj) {
          histVec[histVecIndx++] = sparseConVec[idxVec[i] + jj];
        }
      }
    }
    
    if(nSpksInPrevStep) {
      cudaCheck(cudaMemcpy(dev_histVec, histVec, histVecIndx * sizeof(int), cudaMemcpyHostToDevice));
      callHistogramKernel<histogram_atomic_inc, 1>(dev_histVec, xform, sum, 0, histVecIndx, 0, &histCountI[0], (int)N_NEURONS);
      /*      cudaCheck(cudaMemcpy(dev_histCountI, histCountI, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));*/
      cudaCheck(cudaMemcpyAsync(dev_histCountI, histCountI, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice, stream1));
    }


    /*    expDecay<<<BlocksPerGrid, ThreadsPerBlock>>>();*/

    /*computeConductance<<<BlocksPerGrid, ThreadsPerBlock>>>();*/
    cudaCheck(cudaStreamSynchronize(stream1));
    computeConductanceHist<<<(N_NEURONS + 512 - 1) / 512, 512>>>(dev_histCountE, dev_histCountI);
    cudaCheckLastError("g");
    computeIsynap<<<BlocksPerGrid, ThreadsPerBlock>>>(k*DT);
    cudaCheckLastError("isyp");
  }
  cudaCheck(cudaFreeHost(histVec));
  cudaCheck(cudaFree(dev_histVec));
  cudaCheck(cudaFree(dev_histCountE));
  cudaCheck(cudaFree(dev_histCountI));
  cudaCheck(cudaFreeHost(histCountE));  
  cudaCheck(cudaFreeHost(histCountI));
  fclose(fpIFR);
  cudaCheck(cudaStreamDestroy(stream1));
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
  double curE[N_CURRENT_STEPS_TO_STORE], curI[N_CURRENT_STEPS_TO_STORE], ibgCur[N_CURRENT_STEPS_TO_STORE], *dev_curE, *dev_curI, *dev_ibg, curIff[N_CURRENT_STEPS_TO_STORE], *dev_curiff;
  cudaCheck(cudaGetSymbolAddress((void **)&dev_curE, glbCurE));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_curI, glbCurI));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_ibg, dev_bgCur));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_curiff, dev_iff));
  cudaCheck(cudaMemcpy(curE, dev_curE, N_CURRENT_STEPS_TO_STORE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(curI, dev_curI, N_CURRENT_STEPS_TO_STORE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(ibgCur, dev_ibg, N_CURRENT_STEPS_TO_STORE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(curIff, dev_curiff, N_CURRENT_STEPS_TO_STORE * sizeof(double), cudaMemcpyDeviceToHost));
  printf("\n nSpks = %d\n", *nSpks);
  printf(" MAX SPKS stored on GPU = %d \n", MAX_SPKS); 
  printf("\n Simulation completed ! \n");
  fpElapsedTime = fopen("elapsedTime.csv", "a+");
  fprintf(fpElapsedTime, "%llu %f %d\n", N_NEURONS, elapsedTime, *nSpks);
  fclose(fpElapsedTime);
  /* ================= SAVE TO DISK =============================================================*/

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
  fclose(fpSpkTimes);
  printf("done\n");
  if(IF_SAVE) {  
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
    for(i = 0; i < N_CURRENT_STEPS_TO_STORE; ++i) {
      fprintf(fpCur, "%f;%f;%f;%f\n", curE[i], curI[i], ibgCur[i], curIff[i]);
      /*      fprintf(fpCur, "%f\n", curIff[i]);*/
    }
    fclose(fpCur);
    fpConMat = fopen("conMat.csv", "w");
    fpConMat = fopen("conVec.csv", "w");

    /*    for(i = 0; i < N_NEURONS; ++i) {
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
  cudaCheck(cudaFreeHost(host_IF_SPK));
  cudaCheck(cudaFreeHost(host_prevStepSpkIdx));
  /*  cudaDeviceReset()*/
  return EXIT_SUCCESS;
}

