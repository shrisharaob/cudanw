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
  double tStart = 0.0, tStop = TSTOP;
  double *spkTimes, *vm = NULL, host_theta = 0.0; /* *vstart; 500 time steps */
  int *nSpks, *spkNeuronIds, nSteps, i, k, lastNStepsToStore;
  double *dev_vm = NULL, *dev_spkTimes, *dev_time = NULL, *host_time;
  int *dev_conVec = NULL, *dev_nSpks, *dev_spkNeuronIds;
  FILE *fp, *fpSpkTimes, *fpElapsedTime; // *fpConMat, 
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
  char filetag[16];
  double *firingrate = NULL;
  double host_rewiredEEWeight = 1;
  cudaCheck(cudaStreamCreate(&stream1));

  firingrate = (double *) malloc(sizeof(double) * N_NEURONS);
  
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
  if(argc > 4) {
    strcpy(filetag, argv[4]);
  }
  // if(argc > 5) {
  //     kappa = atof(argv[2]);
  // }
  if(argc > 5) {
      host_rewiredEEWeight = atof(argv[5]);
  }

  cudaMemcpyToSymbol(rewiredEEWeight, &host_rewiredEEWeight, sizeof(host_rewiredEEWeight));

  printf("\n Computing on GPU%d \n", deviceId);
  cudaCheck(cudaSetDevice(deviceId));
  host_theta = PI * host_theta / (180.0); /* convert to radians */
  cudaMemcpyToSymbol(theta, &host_theta, sizeof(host_theta));
  /* ================= INITIALIZE ===============================================*/
  nSteps = (tStop - tStart) / DT;
  lastNStepsToStore = (int)floor(STORE_LAST_T_MILLISEC  / DT);
  //  nSteps = 800;
  printf("\n N  = %llu \n NE = %llu \n NI = %llu \n K  = %d \n tStop = %3.2f seconds nSteps = %d\n\n", N_NEURONS, NE, NI, (int)K, tStop / 1000.0, nSteps);
  
  printf(" theta = %2.3f contrast = %2.1f\n, ksi = %f\n", host_theta, HOST_CONTRAST, ETA_E);
  
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 128;
  int BlocksPerGrid = (N_NEURONS + ThreadsPerBlock - 1) / ThreadsPerBlock;
  printf("Threads per block : %d, Blocks per grid : %d \n", ThreadsPerBlock, BlocksPerGrid);
  /*INITIALIZE RND GENERATORS FOR ibf & iff */
  setupBGCurGenerator<<<BlocksPerGrid, ThreadsPerBlock>>>(time(NULL));
  setupIFFRndGenerator<<<BlocksPerGrid, ThreadsPerBlock>>>(time(NULL));
  /*Generate frozen FF input approximat*/
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  unsigned long long tttt = 45687ULL;
  cudaCheck(cudaMalloc((void **)&devNormRandState, N_NEURONS * sizeof(curandState)));
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, tttt);
  tttt = 12463ULL;
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devNormRandState, tttt);
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
  // cudaCheck(cudaGetSymbolAddress((void **)&dev_sparseVec, dev_sparseConVec));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_idxVec, dev_sparseIdx));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nPostneuronsPtr, dev_nPostNeurons));
  
  // cudaCheck(cudaMallocHost((void **)&sparseConVec, N_NEURONS * (2 * K + 1) * sizeof(int)));
  /*  cudaCheck(cudaMalloc((void **)&dev_sparseVec, N_NEURONS * ((int)2 * K + 1)* sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_idxVec, N_NEURONS * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_nPostneuronsPtr, N_NEURONS * sizeof(int)));*/
  /*  GenSparseMat(conVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);*/
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  fpSparseConVec = fopen("sparseConVec.dat", "rb");
  fpIdxVec = fopen("idxVec.dat", "rb");
  fpNpostNeurons = fopen("nPostNeurons.dat", "rb");
  unsigned long int dummy, nConnections = 0;
  dummy = fread(idxVec, sizeof(*idxVec), N_NEURONS, fpIdxVec);
  dummy = fread(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpIdxVec);
  fclose(fpNpostNeurons);
  
  for(i = 0; i < N_NEURONS; ++i) {
    nConnections += nPostNeurons[i];
  }
  cudaCheck(cudaMallocHost((void **)&sparseConVec, nConnections * sizeof(int)));  
  dummy = fread(sparseConVec, sizeof(*sparseConVec), nConnections, fpSparseConVec);
  fclose(fpSparseConVec);
  printf("\n #connections = %lu, nElements read  = %lu", nConnections, dummy); fflush(stdout);
  if(dummy != nConnections) {
    printf("sparseConvec read error ? \n");
  }
  cudaCheck(cudaMalloc((void **)&dev_sparseVec,  nConnections * sizeof(int)));
  devPtrs.dev_sparseConVec = dev_sparseVec;
  cudaCheck(cudaMemcpy(dev_sparseVec, sparseConVec, nConnections * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_idxVec, idxVec, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_nPostneuronsPtr, nPostNeurons, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
 ////////////////////////////////////////////////////////////////////////////////
  devPtrs.dev_IS_REWIRED_LINK = NULL;
  if(IF_REWIRE) {
    unsigned long nElementsRead = 0;
    FILE *fpRewired;
    unsigned int *IS_REWIRED_LINK = NULL; 
    cudaCheck(cudaMallocHost((void **)&IS_REWIRED_LINK,  nConnections * sizeof(unsigned int)));
    fpRewired = fopen("newPostNeurons.dat", "rb");
    nElementsRead = fread(IS_REWIRED_LINK, sizeof(*IS_REWIRED_LINK), nConnections, fpRewired);
    fclose(fpRewired);
    if(nConnections != nElementsRead) {
      printf("rewired read error ? \n");
    }
    printf("done\n");
    unsigned int *dev_IS_REWIRED_LINKPtr = NULL;
    cudaCheck(cudaMalloc((void **)&dev_IS_REWIRED_LINKPtr,  nConnections * sizeof(unsigned int)));
    devPtrs.dev_IS_REWIRED_LINK = dev_IS_REWIRED_LINKPtr;
    cudaCheck(cudaMemcpy(dev_IS_REWIRED_LINKPtr, IS_REWIRED_LINK, nConnections * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaCheck(cudaFreeHost(IS_REWIRED_LINK));    
  }
  //  devPtrs.dev_sparseConVec = dev_sparseVec;
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
  /*  int nSpksInPrevStep;*/
  cudaCheck(cudaMallocHost((void **)&host_IF_SPK, N_NEURONS * sizeof(int)));
  cudaCheck(cudaMallocHost((void **)&host_prevStepSpkIdx, N_NEURONS * sizeof(int)));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_IF_SPK_Ptr, dev_IF_SPK));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_prevStepSpkIdxPtr, dev_prevStepSpkIdx));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nEPtr, dev_ESpkCountMat));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nIPtr, dev_ISpkCountMat));

  for(i = 0; i < N_NEURONS; ++i) {
    host_IF_SPK[i] = 0;
    firingrate[i] = 0.0;
    // conductanceE[i] = 0.0;
    // conductanceI[i] = 0.0;
  }

  
  /* TIME LOOP */
  size_t sizeOfInt = sizeof(int);
  /* SETUP TIMER EVENTS ON DEVICE */
  cudaEventCreate(&stop0); cudaEventCreate(&start0);
  cudaEventRecord(start0, 0);
  unsigned int spksE = 0, spksI = 0;
  FILE *fpIFR = fopen("instant_fr.csv", "w");
  for(k = 0; k < nSteps; ++k) { 
    /*    cudaCheck(cudaMemsetAsync(dev_nEPtr, 0, N_NEURONS * N_SPKS_IN_PREV_STEP * sizeOfInt, stream1));
	  cudaCheck(cudaMemsetAsync(dev_nIPtr, 0, N_NEURONS * N_SPKS_IN_PREV_STEP * sizeOfInt, stream1));*/
    /*    nSpksInPrevStep = 0;*/
    devPtrs.k = k;
    rkdumbPretty<<<BlocksPerGrid, ThreadsPerBlock>>> (kernelParams, devPtrs);
    cudaCheckLastError("rk");
    cudaCheck(cudaMemcpy(host_IF_SPK, dev_IF_SPK_Ptr, N_NEURONS * sizeOfInt, cudaMemcpyDeviceToHost));
  
    for(i = 0; i < N_NEURONS; ++i) {
      if(host_IF_SPK[i]) {
	if(k * DT >= DISCARDTIME) {
	  firingrate[i] += host_IF_SPK[i];
	}
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
    if(!(k%1000)) {
      fprintf(fpIFR, "%f %f \n", ((double)spksE) / (0.05 * (double)NE), ((double)spksI) / (0.05 * (double)NI));fflush(fpIFR);
      fprintf(stdout, "%f %f \n", ((double)spksE) / (0.05 * (double)NE), ((double)spksI) / (0.05 * (double)NI));
      spksE = 0; 
      spksI = 0;
    }
    /*
    if(nSpksInPrevStep > N_SPKS_IN_PREV_STEP) { 
      printf("\nExceeded N_SPKS_IN_PREV_STEP ! nSpksInPrevStep = %d, step = %d \n", nSpksInPrevStep, k); 
      nSpksInPrevStep = N_SPKS_IN_PREV_STEP;
      /*      exit(-1);
    }*/
    /*    cudaCheck(cudaMemcpy(dev_prevStepSpkIdxPtr, host_prevStepSpkIdx, N_NEURONS * sizeOfInt, cudaMemcpyHostToDevice));*/
    expDecay<<<BlocksPerGrid, ThreadsPerBlock>>>();
    cudaCheckLastError("exp");
    /*    cudaStreamSynchronize(stream1);
    computeG_Optimal<<<BlocksPerGrid, ThreadsPerBlock>>>();
    spkSum<<<BlocksPerGrid, ThreadsPerBlock>>>(nSpksInPrevStep);*/
    computeConductance<<<BlocksPerGrid, ThreadsPerBlock>>>();
    cudaCheckLastError("g");
    computeIsynap<<<BlocksPerGrid, ThreadsPerBlock>>>(k*DT);
    cudaCheckLastError("isyp");
  }
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

  // store firing rates
  char fileSuffix[128], filename[128];
  strcpy(filename, "firingrates");
  double theta_degrees = 180.0 * host_theta / PI;
  // sprintf(fileSuffix, "_xi%1.1f_theta%d_%.2f_%1.1f_cntrst%.1f_%d_tr%s", ETA_E, (int)theta_degrees, ALPHA, TAU_SYNAP_E, HOST_CONTRAST, (int)(tStop),filetag);
  sprintf(fileSuffix, "_xi%1.1f_theta%d_%.2f_%1.1f_cntrst%.1f_%d_tr%s", ETA_E, (int)theta_degrees, 0.0, TAU_SYNAP, HOST_CONTRAST, (int)(tStop), filetag);  
  strcat(filename, fileSuffix);
  FILE *fpFiringrate = fopen(strcat(filename, ".csv"),"w");
  for(i = 0; i < N_NEURONS; ++i) {
    fprintf(fpFiringrate, "%f\n", firingrate[i] / ((tStop - DISCARDTIME) * 0.001));
  }
  fclose(fpFiringrate);
  ////////////////////////////////////////////////////////////////////////////////  
  if(IF_SAVE) {
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
    
    // fpConMat = fopen("conMat.csv", "w");
    // fpConMat = fopen("conVec.csv", "w");

    // /*    for(i = 0; i < N_NEURONS; ++i) {
    // 	  for(k = 0; k < N_NEURONS; ++k) {
    // 	  fprintf(fpConMat, "%d", conVec[i *  N_NEURONS + k]);
    // 	  }
    // 	  fprintf(fpConMat, "\n");

    // 	  }*/
    // fclose(fpConMat);
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
  cudaCheck(cudaFree(dev_sparseVec));
  /* cudaCheck(cudaFree(dev_idxVec));
  cudaCheck(cudaFree(dev_nPostneuronsPtr));*/
  cudaCheck(cudaFreeHost(host_IF_SPK));
  cudaCheck(cudaFreeHost(host_prevStepSpkIdx));
  /*  cudaDeviceReset()*/
  if(IF_REWIRE) {
    cudaCheck(cudaFree(devPtrs.dev_IS_REWIRED_LINK));
  }  
  return EXIT_SUCCESS;
}

