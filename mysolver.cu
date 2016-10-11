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
  double tStart = 0.0, tStop = 55000.0;
  double *spkTimes, *vm = NULL, host_theta = 0.0, theta_degrees; /* *vstart; 500 time steps */
  int *nSpks, *spkNeuronIds, nSteps, i, k, lastNStepsToStore;
  double *dev_vm = NULL, *dev_spkTimes, *dev_time = NULL, *host_time;
  int *dev_conVec = NULL, *dev_nSpks, *dev_spkNeuronIds;
  FILE *fp,  *fpSpkTimes, *fpElapsedTime; // *fpConMat,
  double *host_isynap, *synapticCurrent = NULL;
  /*  int *conVec;*/
  curandState *devStates, *devNormRandState, *poisRandState;;
  cudaEvent_t start0, stop0;
  float elapsedTime;
  int *dev_sparseVec = NULL, *sparseConVec = NULL;
  int *dev_sparseVecFF = NULL, *sparseConVecFF = NULL;
  int idxVec[N_NEURONS], nPostNeurons[N_NEURONS], *dev_idxVec = NULL, *dev_nPostneuronsPtr = NULL;
  int *idxVecFF = NULL, *nPostNeuronsFF = NULL, *dev_idxVecFF = NULL, *dev_nPostneuronsPtrFF = NULL;
  int deviceId = 0;
  devPtr_t devPtrs;
  kernelParams_t kernelParams;
  int IF_SAVE = 1;
  cudaStream_t stream1;
  char filetag[16];
  double *firingrate;
  // double tStep = 1000; // 1000ms
  // char frfilename[128];

  firingrate = (double *) malloc(sizeof(double) * N_NEURONS);
  
  
  cudaCheck(cudaStreamCreate(&stream1));
  printf("old tstop = %f\n", tStop);
  idxVecFF = (int *)malloc((unsigned long long)NFF * sizeof(int));
  nPostNeuronsFF = (int *)malloc((unsigned long long)NFF * sizeof(int));
  //  cudaCheck(cudaMalloc((void **)&idxVecFF,  NFF * sizeof(*idxVecFF)));
  //  cudaCheck(cudaMalloc((void **)&nPostNeuronsFF,  NFF * sizeof(*nPostNeuronsFF)));
  /*PARSE INPUTS*/
  if(argc > 1) {
    deviceId = atoi(argv[1]);
    if(argc > 2) {
      IF_SAVE = atoi(argv[2]);
    }
    if(argc > 3) {
      host_theta = atof(argv[3]);
    }
    if(argc > 4) {
      //      tStop = tStop + atof(argv[4]);
      strcpy(filetag, argv[4]);
    }
  }
  printf("\n Computing on GPU%d \n", deviceId);
  cudaCheck(cudaSetDevice(deviceId));
  theta_degrees = host_theta;
  host_theta = PI * host_theta / (180.0); /* convert to radians */
  /* ================ SIMULATING SOME JITTER IN INPUT ORIENTATION (COULD BE DUE TO EYE TITLT, TO CHECK FANO FACTOR M TUNING) ======= */
  //srand(time(NULL));
  //  double tmprnd = ((double) rand() / (RAND_MAX + 1.0)) * (5.0) - (2.5); // simulatinge eye tilt
  //  host_theta += (tmprnd * PI / 180.0);
  
  cudaMemcpyToSymbol(theta, &host_theta, sizeof(host_theta));
  /* ================= INITIALIZE ===============================================*/
  nSteps = (tStop - tStart) / DT;
  lastNStepsToStore = (int)floor(STORE_LAST_T_MILLISEC  / DT);
  //  nSteps = 800;
  printf("\n N  = %llu \n NE = %llu \n NI = %llu \n K  = %d \n tStop = %d milli seconds nSteps = %d\n\n", N_NEURONS, NE, NI, (int)K, (int)tStop, nSteps);
  
  printf(" theta = %2.3f \n contrast = %2.1f\n ksi = %f\n dt = %f \n tau = %f \n EXP_SUM = %.16f\n", host_theta * 180.0 / PI, HOST_CONTRAST, ETA_E, DT, TAU_SYNAP, EXP_SUM);
  printf("alpha = %f, RHO = %f\n", ALPHA, RHO);
  
  // STORE MEAN G_FF
  double *dev_GFFmeanPtr = NULL, *host_GFFmean = NULL; 
  unsigned long long *devGFFCounterPtr = NULL, *hostGFFCounter = NULL;
  //  cudaCheck(cudaMalloc((void **)&dev_GFFmean,  N_NEURONS * sizeof(*dev_GFFmean)));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_GFFmeanPtr, dev_GFFmean));
  cudaCheck(cudaGetSymbolAddress((void **)&devGFFCounterPtr, devGFFCounter));
  cudaCheck(cudaMallocHost((void **)&host_GFFmean,  N_NEURONS * sizeof(*host_GFFmean)));
  cudaCheck(cudaMallocHost((void **)&hostGFFCounter, sizeof(*hostGFFCounter)));
  *hostGFFCounter = 0;


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
  tttt = 12463ULL; //this has to be fixed so that the input structure is kept the same when running multiple simulations, or else each realization will give different tuning curvs 
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devNormRandState, tttt);
  AuxRffTotal<<<BlocksPerGrid, ThreadsPerBlock>>>(devNormRandState, devStates);
  cudaCheck(cudaFree(devNormRandState));
  /* SETUP POISSION RAND GENERATOR */
  cudaCheck(cudaMalloc((void **)&poisRandState,  NFF * sizeof(curandState)));
  setup_pois_kernel<<<(NFF + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(poisRandState, time(NULL));
  AuxRffTotalWithOriMap<<<(NFF + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(); // assign POs in the ori map for the poiss generators 
  /* ===================================================================================================================== */
  /* SAVE ORIMAP IN LAYER 4 */
  float *dev_L4PO_Ptr = NULL, *host_L4PO = NULL;
  cudaCheck(cudaGetSymbolAddress((void **)&dev_L4PO_Ptr, POInOriMap));
  cudaCheck(cudaMallocHost((void **)&host_L4PO, NFF * sizeof(float)));
  cudaCheck(cudaMemcpy(host_L4PO, dev_L4PO_Ptr, NFF * sizeof(float), cudaMemcpyDeviceToHost));
  FILE *fpPo = fopen("poinmap.csv", "w");
  for(int ikl = 0; ikl < NFF; ikl++) {
    fprintf(fpPo, "%f\n", host_L4PO[ikl]);
  }
  fclose(fpPo);
  cudaCheck(cudaFreeHost(host_L4PO));
  /* ===================================================================================================================== */
  //  cudaCheck(cudaGetSymbolAddress((void **)&dev_sparseVec, dev_sparseConVec));
  int *dev_sparseConVecPtr = NULL, *dev_sparseConVecFFPtr = NULL;
  cudaCheck(cudaGetSymbolAddress((void **)&dev_idxVec, dev_sparseIdx));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nPostneuronsPtr, dev_nPostNeurons));
  //  cudaCheck(cudaMallocHost((void **)&sparseConVec, N_NEURONS * (2 * K + 1) * sizeof(int)));
  //  cudaCheck(cudaGetSymbolAddress((void **)&dev_sparseVecFF, dev_sparseConVecFF));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_idxVecFF, dev_sparseIdxFF));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nPostneuronsPtrFF, dev_nPostNeuronsFF));
  //  cudaCheck(cudaMallocHost((void **)&sparseConVecFF, N_NEURONS * (2ULL * (unsigned long long)(CFF * K) + 1) * sizeof(int)));
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  fpSparseConVec = fopen("sparseConVec.dat", "rb");
  fpIdxVec = fopen("idxVec.dat", "rb");
  fpNpostNeurons = fopen("nPostNeurons.dat", "rb");
  int dummy;
  dummy = fread(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpNpostNeurons);
  unsigned long int nConnections = 0;
  for(i = 0; i < N_NEURONS; ++i) {
    nConnections += nPostNeurons[i];
  }
  cudaCheck(cudaMallocHost((void **)&sparseConVec, nConnections * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_sparseConVecPtr,  nConnections * sizeof(int)));
  devPtrs.dev_sparseConVec = dev_sparseConVecPtr;
  //  dummy = fread(sparseConVec, sizeof(*sparseConVec), N_NEURONS * (2 * (int)K + 1), fpSparseConVec);
  dummy = fread(sparseConVec, sizeof(*sparseConVec), nConnections, fpSparseConVec);
  fclose(fpSparseConVec);
  if(dummy != nConnections) {
    printf("sparseConvec read error ? \n");
  }

  dummy = fread(idxVec, sizeof(*idxVec), N_NEURONS, fpIdxVec);
  fclose(fpIdxVec);
  /* READ FF SPARSE CONNECTION MAT */
  FILE *fpSparseConVecFF, *fpIdxVecFF, *fpNpostNeuronsFF;
  fpSparseConVecFF = fopen("sparseConVecFF.dat", "rb");
  fpIdxVecFF = fopen("idxVecFF.dat", "rb");
  fpNpostNeuronsFF = fopen("nPostNeuronsFF.dat", "rb");
  dummy = fread(nPostNeuronsFF, sizeof(*nPostNeuronsFF), NFF, fpNpostNeuronsFF);
  fclose(fpNpostNeuronsFF);
  unsigned long int nConnectionsFF = 0;
  for(i = 0; i < NFF; ++i) {
    nConnectionsFF += nPostNeuronsFF[i];
  }
  cudaCheck(cudaMalloc((void **)&dev_sparseConVecFFPtr,  nConnectionsFF * sizeof(int)));
  cudaCheck(cudaMallocHost((void **)&sparseConVecFF, nConnectionsFF * sizeof(int)));

  devPtrs.dev_sparseConVecFF = dev_sparseConVecFFPtr;
  //  dummy = fread(sparseConVecFF, sizeof(*sparseConVecFF), N_NEURONS * (2ULL * (unsigned long long)(CFF * K) + 1), fpSparseConVecFF);
  dummy = fread(sparseConVecFF, sizeof(*sparseConVecFF), nConnectionsFF, fpSparseConVecFF);
  fclose(fpSparseConVecFF);
  if(dummy != nConnectionsFF) {
    printf("%d, sparseConvecFF read error ? \n", dummy);
  }
  dummy = fread(idxVecFF, sizeof(*idxVecFF), NFF, fpIdxVecFF);
  fclose(fpIdxVecFF);
  //  cudaCheck(cudaMemcpy(dev_sparseVec, sparseConVec, N_NEURONS * (2 * K + 1) * sizeof(int), cudaMemcpyHostToDevice));
  printf("sparse convec ptr: %p\n", dev_idxVec);
  cudaCheck(cudaMemcpy(dev_sparseConVecPtr, sparseConVec,  nConnections * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_idxVec, idxVec, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_nPostneuronsPtr, nPostNeurons, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
  printf("tst point, %p ----- %p \n", dev_sparseVec, dev_sparseVecFF);
  //  cudaCheck(cudaMemcpy(dev_sparseVecFF, sparseConVecFF, N_NEURONS * (2ULL * (unsigned long long)(CFF * K) + 1)* sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_sparseConVecFFPtr, sparseConVecFF, nConnectionsFF * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_idxVecFF, idxVecFF, NFF * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_nPostneuronsPtrFF, nPostNeuronsFF, NFF * sizeof(int), cudaMemcpyHostToDevice));
 /* ================= ALLOCATE PAGELOCKED MEMORY ON HOST =========================*/
  cudaCheck(cudaMallocHost((void **)&spkTimes, MAX_SPKS  * sizeof(*spkTimes)));
  cudaCheck(cudaMallocHost((void **)&host_isynap, N_I_SAVE_CUR * sizeof(*host_isynap)));
  cudaCheck(cudaMallocHost((void **)&vm,  lastNStepsToStore * N_NEURONS_TO_STORE * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&host_time,  lastNStepsToStore * sizeof(*vm)));
  cudaCheck(cudaMallocHost((void **)&nSpks, sizeof(*nSpks)));
  cudaCheck(cudaMallocHost((void **)&spkNeuronIds, MAX_SPKS * sizeof(*spkNeuronIds)));
  /* ================= ALLOCATE GLOBAL MEMORY ON DEVICE ===========================*/
  /*cudaCheck(cudaMalloc((void **)&dev_conVec, N_NEURONS * N_NEURONS * sizeof(int)));*/
  cudaCheck(cudaMalloc((void **)&dev_vm, lastNStepsToStore * N_NEURONS_TO_STORE * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&dev_time, lastNStepsToStore * sizeof(double)));
  cudaCheck(cudaMalloc((void **)&synapticCurrent, N_I_SAVE_CUR * N_NEURONS * sizeof(double)));
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
  int *dev_IF_SPK_Ptr = NULL, *dev_prevStepSpkIdxPtr = NULL, *host_IF_SPK = NULL, *host_prevStepSpkIdx = NULL,  *dev_nEPtr = NULL, *dev_nIPtr = NULL, *dev_IF_SPK_POISSION_Ptr = NULL;
  int *host_FF_IF_SPK, nFFSpksInPrevStep;
  int nSpksInPrevStep;
  cudaCheck(cudaMallocHost((void **)&host_IF_SPK, N_NEURONS * sizeof(int)));
  cudaCheck(cudaMallocHost((void **)&host_prevStepSpkIdx, N_NEURONS * sizeof(int)));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_IF_SPK_Ptr, dev_IF_SPK));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_prevStepSpkIdxPtr, dev_prevStepSpkIdx));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nEPtr, dev_ESpkCountMat));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_nIPtr, dev_ISpkCountMat));
  cudaCheck(cudaGetSymbolAddress((void **)&dev_IF_SPK_POISSION_Ptr, IF_SPIKE_POISSION_SPK));
  cudaCheck(cudaMallocHost((void **)&host_FF_IF_SPK, NFF * sizeof(int)));
  printf("\n %p \n", host_FF_IF_SPK);
  cudaCheck(cudaMemcpy(host_FF_IF_SPK, dev_IF_SPK_POISSION_Ptr, NFF * sizeof(int), cudaMemcpyDeviceToHost));
  for(i = 0; i < N_NEURONS; ++i) {
    host_IF_SPK[i] = 0;
    firingrate[i] = 0.0;
    if(i < NFF) {
      host_FF_IF_SPK[i] = 0;
    }
  }
  /* TIME LOOP */
  size_t sizeOfInt = sizeof(int);
  size_t sizeOfDbl = sizeof(double);
  /* SETUP TIMER EVENTS ON DEVICE */
  cudaEventCreate(&stop0); cudaEventCreate(&start0);
  cudaEventRecord(start0, 0);
  unsigned int spksE = 0, spksI = 0, spksFF = 0;
  FILE *fpIFR = fopen("instant_fr.csv", "w");
  int *histVec = NULL, *dev_histVec = NULL, *histVecFF = NULL, *dev_histVecFF = NULL; /* for storing the post-synaptic neurons to be updated */
  int histVecIndx = 0, histVecIndxFF = 0;
  unsigned int histVecLength = N_NEURONS * (int)K;
  unsigned int histVecLengthFF = N_NEURONS * (int)(CFF * K);
  if((unsigned long long)K >= NE | (unsigned long long)K >= NI) {
    histVecLength = (unsigned int)(N_NEURONS * N_NEURONS);
  }
  cudaCheck(cudaMallocHost((void **)&histVec, histVecLength * sizeof(*histVec)));
  cudaCheck(cudaMalloc((void **)&dev_histVec, histVecLength * sizeof(*dev_histVec)));

  cudaCheck(cudaMallocHost((void **)&histVecFF, histVecLengthFF * sizeof(*histVecFF)));
  cudaCheck(cudaMalloc((void **)&dev_histVecFF, histVecLengthFF * sizeof(*dev_histVecFF)));

  test_xform xform; // defined in cuda_histogram.h
  test_sumfun sum;  // defined in cuda_histogram.h
  int *dev_histCountE = NULL, *histCountE = NULL, *dev_histCountI = NULL, *histCountI = NULL;
  int *dev_histCountFF = NULL, *histCountFF = NULL;
  cudaCheck(cudaMalloc((void **)&dev_histCountE, sizeof(int) * N_NEURONS));
  cudaCheck(cudaMallocHost((void **)&histCountE, sizeof(int) * N_NEURONS));
  cudaCheck(cudaMalloc((void **)&dev_histCountI, sizeof(int) * N_NEURONS));
  cudaCheck(cudaMallocHost((void **)&histCountI, sizeof(int) * N_NEURONS));

  cudaCheck(cudaMalloc((void **)&dev_histCountFF, sizeof(int) * N_NEURONS));
  cudaCheck(cudaMallocHost((void **)&histCountFF, sizeof(int) * N_NEURONS));

  int tmp;
  char fileSuffix[128], filename[128];
  strcpy(filename, "currents");
  sprintf(fileSuffix, "_%1.1f_%1.f", ALPHA, TAU_SYNAP);
  strcat(filename, fileSuffix);
  FILE *fpCur = NULL;
  fpCur = fopen(strcat(filename, ".csv"), "w");
  /*printf("\n\n\n\n %d\n\n\n\n", sparseConVec[835584ULL]);*/


  for(k = 0; k < nSteps; ++k) { 
    /*    cudaCheck(cudaMemsetAsync(dev_nEPtr, 0, N_NEURONS * N_SPKS_IN_PREV_STEP * sizeOfInt, stream1));
	  cudaCheck(cudaMemsetAsync(dev_nIPtr, 0, N_NEURONS * N_SPKS_IN_PREV_STEP * sizeOfInt, stream1));*/
    /*    nSpksInPrevStep = 0;*/
    devPtrs.k = k;
    nSpksInPrevStep = 0;
    histVecIndx = 0;
    histVecIndxFF = 0;
    for(i = 0; i < N_NEURONS; ++i) {
      histCountI[i] = 0;
      histCountE[i] = 0;
      histCountFF[i] = 0;
    }

    GenPoissionSpikeInFFLayer<<<(NFF + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(poisRandState); // GENERATE SPIKES IN LAYER 4 
    rkdumbPretty<<<BlocksPerGrid, ThreadsPerBlock>>> (kernelParams, devPtrs);
    cudaCheckLastError("rk");
    if(k > 0) {
      /*      cudaCheck(cudaMemcpy(host_IF_SPK, dev_IF_SPK_Ptr, N_NEURONS * sizeOfInt, cudaMemcpyDeviceToHost));*/
      cudaCheck(cudaMemcpyAsync(host_IF_SPK, dev_IF_SPK_Ptr, N_NEURONS * sizeOfInt, cudaMemcpyDeviceToHost, stream1));
      cudaCheck(cudaMemcpyAsync(host_isynap, synapticCurrent, N_I_SAVE_CUR * sizeOfDbl, cudaMemcpyDeviceToHost, stream1));
      cudaCheck(cudaMemcpyAsync(host_FF_IF_SPK, dev_IF_SPK_POISSION_Ptr, NFF * sizeOfInt, cudaMemcpyDeviceToHost, stream1));
    }
    cudaCheck(cudaStreamSynchronize(stream1));
    /*instantaneous firing rate, rect non-overlapping window */
    for(i = 0; i < N_NEURONS; ++i) {
      if(k * DT > DISCARDTIME) {
	firingrate[i] += host_IF_SPK[i];
      }
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

      if(i < NFF) {
        if(host_FF_IF_SPK[i]) {
            spksFF += 1;
        }
      }
    }
    if(!(k%(int)(50.0/DT))) {
      fprintf(fpIFR, "%f %f \n", ((double)spksE) / (0.05 * (double)NE), ((double)spksI) / (0.05 * (double)NI));fflush(fpIFR);
      fprintf(stdout, "%f %f ", ((double)spksE) / (0.05 * (double)NE), ((double)spksI) / (0.05 * (double)NI));
      spksE = 0; 
      spksI = 0;
      fprintf(stdout, "%f \n", ((double)spksFF) / (0.05 * (double)NFF));
      spksFF= 0;
    }
    // if(!(fmod((k * DT ), tStep))) {
    //   FILE *fpFrChunk;
    //   //      fpFrChunk = sprintf(frfilename, "fr_%1.1f_te%1.f_ti%1.f", ALPHA, TAU_SYNAP_E, TAU_SYNAP_I);
    //   sprintf(frfilename, "fr_xi%1.1f_theta%d_%.2f_%1.1f_cntrst%.1f_%d_tr%s_tStop%d.csv", ETA_E, (int)theta_degrees, ALPHA, TAU_SYNAP_E, HOST_CONTRAST, (int)(tStop),filetag, (int)(k * DT));
    //   fpFrChunk = fopen(frfilename, "w");
    //   for(i = 0; i < N_NEURONS; ++i) {
    // 	fprintf(fpFrChunk, "%f\n", firingrate[i] / (((k * DT) - DISCARDTIME) * 0.001));
    //   }
    //   fclose(fpFrChunk);
    // }
    /*-----------------------------------------------------------------------*/
    expDecay<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_histCountE, dev_histCountI);
    expDecayGFF<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_histCountFF);
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
      cudaCheckLastError("HIST");
      cudaCheck(cudaMemcpy(dev_histCountI, histCountI, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
      /*      cudaCheck(cudaMemcpyAsync(dev_histCountI, histCountI, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice, stream1));*/
    }

    histVecIndxFF = 0;
    nFFSpksInPrevStep = 0;
    for(i = 0; i < NFF; ++i) {
      if(host_FF_IF_SPK[i]){
        nFFSpksInPrevStep += 1;
        for(int jj = 0; jj < nPostNeuronsFF[i]; ++jj) {
          histVec[histVecIndxFF++] = sparseConVecFF[idxVecFF[i] + jj];
        }
      }
    }
    if(nFFSpksInPrevStep) {
      cudaCheck(cudaMemcpy(dev_histVecFF, histVec, histVecIndxFF * sizeof(int), cudaMemcpyHostToDevice));
      callHistogramKernel<histogram_atomic_inc, 1>(dev_histVecFF, xform, sum, 0, histVecIndxFF, 0, &histCountFF[0], (int)N_NEURONS);
      cudaCheckLastError("HIST_FF");
      cudaCheck(cudaMemcpy(dev_histCountFF, histCountFF, N_NEURONS * sizeof(int), cudaMemcpyHostToDevice));
    }

    // int tmpCnt = 0;
    // for(i = 0; i < N_NEURONS; ++i) {
    //   if(histCountFF[i]) {
    //     tmpCnt += histCountFF[i];
    //   }
    // }
    //    printf("histcount = %d\n", tmpCnt);
    /*    expDecay<<<BlocksPerGrid, ThreadsPerBlock>>>();*/

    /*computeConductance<<<BlocksPerGrid, ThreadsPerBlock>>>();*/
    cudaCheck(cudaStreamSynchronize(stream1));

    /* SAVE CURRENT VALUES TO DISK  */
    // for(int jj = 0; jj < N_I_SAVE_CUR; ++jj) {
    //   fprintf(fpCur, "%f ", host_isynap[jj]);
    // }
    // fprintf(fpCur, "\n";)
    computeConductanceHist<<<(N_NEURONS + 512 - 1) / 512, 512>>>(dev_histCountE, dev_histCountI);
    computeConductanceHistFF<<<(N_NEURONS + 512 - 1) / 512, 512>>>(dev_histCountFF);
    cudaCheckLastError("g");
    computeIsynap<<<BlocksPerGrid, ThreadsPerBlock>>>(k*DT);
    cudaCheckLastError("isyp");
  }
  fclose(fpCur);
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
  cudaCheck(cudaMemcpy(vm, dev_vm, lastNStepsToStore * N_NEURONS_TO_STORE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(host_time, dev_time, lastNStepsToStore * sizeof(double), cudaMemcpyDeviceToHost));
  /*  cudaCheck(cudaMemcpy(host_isynap, synapticCurrent, lastNStepsToStore * N_NEURONS * sizeof(double), cudaMemcpyDeviceToHost));*/
  /*  cudaCheck(cudaMemcpy(vm, dev_vm, lastNStepsToStore * N_NEURONS * sizeof(double), cudaMemcpyDeviceToHost));*/
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
  //  char fileSuffix[128], filename[128];

  strcpy(filename, "spkTimes");
  sprintf(fileSuffix, "_xi%1.1f_theta%d_%.2f_%1.1f_%d_tr%s", ETA_E, (int)theta_degrees, ALPHA, TAU_SYNAP, (int)(tStop), filetag);
  strcat(filename, fileSuffix);
  fpSpkTimes = fopen(strcat(filename, ".csv"),"w");
  /*  fpSpkTimes = fopen("spkTimes.csv", "w");*/
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
  cudaCheck(cudaMemcpy(host_GFFmean, dev_GFFmeanPtr, N_NEURONS * sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(hostGFFCounter, devGFFCounterPtr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  printf("saving vm to disk ....");
  fflush(stdout);
  
  // if(nSteps > 20000) {
  //     strcpy(filename, "gffmean");
  //     sprintf(fileSuffix, "_R0%1.1f_theta%d_%.2f_%1.1f_%d_tr%s", R0, (int)theta_degrees, ALPHA, TAU_SYNAP, (int)(tStop), filetag);
  //     strcat(filename, fileSuffix);
  //     FILE* fpGFFmean;
  //     fpGFFmean = fopen(strcat(filename, ".csv"),"w");
  //     double denom = 0.0;
  //     denom = (double)hostGFFCounter[0];
  //     for(k = 0; k < N_NEURONS; ++k) {
  // 	fprintf(fpGFFmean, "%f\n", (double)host_GFFmean[k] / denom);
  //     }
  //     fclose(fpGFFmean);
  //   }

  printf("computing firing rates ....");
  fflush(stdout);
  strcpy(filename, "firingrates");
  sprintf(fileSuffix, "_xi%1.1f_theta%d_%.2f_%1.1f_cntrst%.1f_%d_tr%s", ETA_E, (int)theta_degrees, ALPHA, TAU_SYNAP, HOST_CONTRAST, (int)(tStop),filetag);
  strcat(filename, fileSuffix);
  FILE *fpFiringrate = fopen(strcat(filename, ".csv"),"w");
  for(i = 0; i < N_NEURONS; ++i) {
    fprintf(fpFiringrate, "%f\n", firingrate[i] / ((tStop - DISCARDTIME) * 0.001));
  }
  fclose(fpFiringrate);


  if(IF_SAVE) {
    /*  SAVE CONDUCTANCES */
    // if(nSteps > 20000) {
    //   strcpy(filename, "gffmean");
    //   sprintf(fileSuffix, "_R0%1.1f_theta%d_%.2f_%1.1f_%d_tr%s", R0, (int)theta_degrees, ALPHA, TAU_SYNAP, (int)(tStop), filetag);
    //   strcat(filename, fileSuffix);
    //   FILE* fpGFFmean;
    //   fpGFFmean = fopen(strcat(filename, ".csv"),"w");
    //   double denom = 0.0;
    //   denom = (double)hostGFFCounter[0];
    //   for(k = 0; k < N_NEURONS; ++k) {
    // 	fprintf(fpGFFmean, "%f\n", (double)host_GFFmean[k] / denom);
    //   }
    //   fclose(fpGFFmean);
    // }
    
//    char fileSuffix[128], filename[128];
    strcpy(filename, "vm");
    sprintf(fileSuffix, "_xi%1.1f_theta%d_%.2f_%1.1f_%d_tr%s", ETA_E, (int)theta_degrees, ALPHA, TAU_SYNAP, (int)(tStop), filetag);
    //sprintf(fileSuffix, "_%1.1f_%1.1f", ALPHA, TAU_SYNAP);
    strcat(filename, fileSuffix);
    fp = fopen(strcat(filename, ".csv"),"w");
    //    fp = fopen("vm.csv", "w");
    for(i = 0; i < lastNStepsToStore; ++i) {
      fprintf(fp, "%f ", host_time[i]);
      for(k = 0; k < N_NEURONS_TO_STORE; ++k) {
	/*	fprintf(fp, "%f %f ", vm[k + i *  N_NEURONS], host_isynap[k + i * N_NEURONS]);*/
        fprintf(fp, "%f ", vm[k + i *  N_NEURONS_TO_STORE]);
      }
      fprintf(fp, "\n");
    }
    printf("\n%d %d\n", i, k);
    fclose(fp);
    FILE* fpCur = fopen("currents.csv", "w");
    for(i = 0; i < N_CURRENT_STEPS_TO_STORE; ++i) {
      fprintf(fpCur, "%f;%f;%f;%f\n", curE[i], curI[i], ibgCur[i], curIff[i]);
    //      fprintf(fpCur, "%f\n", curIff[i]);
    }
    fclose(fpCur);
    // fpConMat = fopen("conMat.csv", "w");
    // fpConMat = fopen("conVec.csv", "w");

    /*    for(i = 0; i < N_NEURONS; ++i) {
      for(k = 0; k < N_NEURONS; ++k) {
	fprintf(fpConMat, "%d", conVec[i *  N_NEURONS + k]);
      }
            fprintf(fpConMat, "\n");

      }*/
    // fclose(fpConMat);
  }
  printf("done\n");
  /*================== CLEANUP ===================================================================*/
  cudaCheck(cudaFreeHost(vm));
  cudaCheck(cudaFreeHost(host_time));
  /*  cudaCheck(cudaFreeHost(host_isynap));*/
  cudaCheck(cudaFreeHost(spkTimes));
  cudaCheck(cudaFreeHost(spkNeuronIds));
  cudaCheck(cudaFreeHost(nSpks));
  cudaCheck(cudaFree(dev_vm));
  cudaCheck(cudaFree(dev_time));
  /*  cudaCheck(cudaFree(synapticCurrent));*/
  cudaCheck(cudaFree(dev_spkNeuronIds));
  cudaCheck(cudaFree(dev_spkTimes));
  cudaCheck(cudaFree(dev_nSpks));
  cudaCheck(cudaFree(devStates));
  /*  cudaCheck(cudaFree(dev_sparseVec));
  cudaCheck(cudaFree(dev_idxVec));
  cudaCheck(cudaFree(dev_nPostneuronsPtr));*/
  cudaCheck(cudaFreeHost(host_IF_SPK));
  cudaCheck(cudaFreeHost(host_prevStepSpkIdx));

  cudaCheck(cudaFreeHost(host_GFFmean));
  cudaCheck(cudaFreeHost(hostGFFCounter));

  free(idxVecFF);
  free(nPostNeuronsFF);
  // cudaCheck(cudaFreeHost(idxVecFF));
  // cudaCheck(cudaFreeHost(nPostNeuronsFF));
  /*  cudaDeviceReset()*/
  free(firingrate);
  return EXIT_SUCCESS;
}

