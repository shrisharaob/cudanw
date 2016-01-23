#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mycurand.h"
#include <stdio.h>
#include "devHostConstants.h"
#include "GenSparseMat.cu"
#include "GenConProbDistDepMat.cu"
#include "tinyRNG.cu"

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

__global__ void initConVec(float *dev_conVec, int maxNeurons) {
  unsigned int mNeuron = threadIdx.x + blockIdx.x * blockDim.x;
  /*  int stride = gridDim.x * blockDim.x;*/
  unsigned long int i;
  if(mNeuron < maxNeurons) {
    for(i = 0; i < N_NEURONS; ++i) {
      dev_conVec[mNeuron + maxNeurons * i] = 0;
    }
    /*  mNeuron += stride;*/
  }
}

__global__ void setup_kernel(curandState *state, unsigned long long seed ) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    if(id < N_NEURONS) {
      curand_init(seed * (id + 7), id, 0, &state[id]);
    }
}

__device__ float randkernel(curandState *state, unsigned long int kNeuron) {
  /*RETURNS ONE SAMPLE FROM UNIFORM DISTRIBUTION*/
  /*  unsigned int id = (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;*/
  float randNumber= 0.0;
  if(kNeuron < N_NEURONS) {
    curandState localState = state[kNeuron]; /* state in global memory */
    randNumber = curand_uniform(&localState);
    state[kNeuron] = localState;
  }
  return randNumber;
}

/*__global__ kernelGenConMat0();*/

__global__ void kernelGenConMat(curandState *state, float *dev_conVec, int lChunck, int maxNeurons){
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  float k, n;
  if(id < maxNeurons & kNeuron < N_NEURONS) {
    k = (float)K;
    /* E --> EI */
    if(kNeuron < NE) {
      n = (float)NE;
    }
    else {
      n = (float)NI;
    }
    for(i = 0; i < N_NEURONS; ++i) {
      // i is row and id is clmn
      if(i < NE) {  /* E --> E/I */
	//        n = (float)NE;
        if(k/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
	  //	  dev_conVec[id + i * maxNeurons] = 1;
          dev_conVec[i + id * maxNeurons] = 1;
        }
      }
      if(i >= NE) { /* I --> E/I */
	//	n = (float)NI;
        if(k/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVec[i + id * maxNeurons] = 1; // dev_conVec[id + i * maxNeurons] = 1;
        } 
      }
    }
    /*    }*/
    /* I --> EI */
    /*
      if(id >= NE & NI > 0) {
      n = (float)NI;
      for(i = 0; i < N_NEURONS; ++i) {
      if(i < NE) {  /* I --> E 
      if(k/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? 
      dev_conVec[id + i * maxNeurons] = 1;
      } 
      }
      if(i >= NE) { /* I --> I 
      if(k/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? 
      dev_conVec[id + i * maxNeurons] = 1;
      } 
      }
      }      }*/
  }
}

//-----------------------------
__global__ void kernelGenConMatWithDiffK(curandState *state, float *dev_conVec, int lChunck, int maxNeurons){
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  float k;
  if(id < maxNeurons & kNeuron < N_NEURONS) {
    if(kNeuron < NE) {
      k = ((float)K) * K_REC_E_PREFACTOR;
    }
    else {
      k = ((float)K) * K_REC_I_PREFACTOR;
    }
    for(i = 0; i < N_NEURONS; ++i) {
      if(i < NE) {  /* E --> E/I */
        if(k/((float)NE) >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
          // dev_conVec[i + id * maxNeurons] = 1;
          dev_conVec[i + id * N_NEURONS] = 1;	  
        }
      }
      if(i >= NE) { /* I --> E/I */
        if(k/((float)NI) >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
	  //	  dev_conVec[i + id * maxNeurons] = 1; //dev_conVec[id + i * maxNeurons] = 1;
          dev_conVec[i + id * N_NEURONS] = 1;	  
        } 
      }
    }
    //======================================
    // for(i = 0; i < N_NEURONS; ++i) {
    //   dev_conVec[i + id * N_NEURONS]  = i + id * N_NEURONS + lChunck * maxNeurons * N_NEURONS;
    // }
    //=========================================
  }
}

//------------------------------
__global__ void kernelFixedEII(curandState *fixedState, curandState *state, float *dev_conVec, int lChunck, int maxNeurons){
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  float k, n;
  curandState *threadState;
  if(id < maxNeurons & kNeuron < N_NEURONS) {
    k = (float)K;
    if(kNeuron < NE) {
      n = (float)NE;
    }
    else {
      n = (float)NI;
    }
    for(i = 0; i < N_NEURONS; ++i) { 
      if(i < NE && kNeuron < NE) { /* E --> E */
	threadState = state; // bad practice !!!!
      }
      else {
	threadState = fixedState;
      }
      if(i < NE) {  
	if(k/n >= randkernel(threadState, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVec[id + i * maxNeurons] = 1;
        }
      }
      if(i >= NE) { /* E --> I */
        if(k/n >= randkernel(threadState, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVec[id + i * maxNeurons] = 1;
        } 
      }
    }
  }
}


__global__ void kernelGenConMatSparseE2E(curandState *state, float *dev_conVec, int lChunck, int maxNeurons){
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  float k, n;
  if(id < maxNeurons & kNeuron < N_NEURONS) {
    k = (float)K;
    /* E --> EI */
    /*    if(id < N_NEURONS & NE > 0) {*/
    if(kNeuron < NE) {
      n = (float)NE;
    }
    else {
      n = (float)NI;
    }
    for(i = 0; i < N_NEURONS; ++i) {
      if(i < NE) {  /* E --> E */
        if(id < NE) {
          if(sqrt(k)/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
	    dev_conVec[id + i * maxNeurons] = 1;
          }
        }
        else {
         if(k/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
	   dev_conVec[id + i * maxNeurons] = 1;
          }
        }
      }
      if(i >= NE) { /* E --> I */
        if(k/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVec[id + i * maxNeurons] = 1;
        } 
      }
    }
  }
}

__global__ void KernelGenDistDepConMat(curandState *state, float *dev_conVec, int lChunck, int maxNeurons){
  /* GENERATE CONNECTION MATRIX WITH ANOTOMIC CONNECTIVITY PROFILE */
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  if(id < maxNeurons & kNeuron < N_NEURONS) {
    /*    k = (float)K;
    if(kNeuron < NE) {
      n = (float)NE;
    }
    else {
      n = (float)NI;
    }*/
    for(i = 0; i < N_NEURONS; ++i) {
      if(i < NE) {  /* E --> E/I */
        if(dev_conVec[id + i * maxNeurons] >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVec[id + i * maxNeurons] = 1;
        }
        else{
          dev_conVec[id + i * maxNeurons] = 0;
        }
      }
      if(i >= NE) { /* I --> E/I */
        if(dev_conVec[id + i * maxNeurons] >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVec[id + i * maxNeurons] = 1;
        } 
        else{
          dev_conVec[id + i * maxNeurons] = 0;
        }
      }
    }
  }
}

__global__ void KernelGenConmatBiDir(curandState *state, float *dev_conVec, int lChunck, int maxNeurons){
  /* GENERATE CONNECTION MATRIX WITH ANOTOMIC CONNECTIVITY PROFILE */
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  double alpha = ALPHA, pBi, pUni, p, k, n;

  if(id < maxNeurons & kNeuron < N_NEURONS) {
    // k = (double)K;
    if(kNeuron < NE) {
      k = ((float)K) * K_REC_E_PREFACTOR;
    }
    else {
      k = ((float)K) * K_REC_I_PREFACTOR;
    }
    if(kNeuron < NE) {
      n = (double)NE;
    }
    else {
      n = (double)NI;
    }
    p = k / n;
    pBi = alpha * p + (1 - alpha) * p * p;
    pUni = (1 - alpha) * p * (1 - p);
    
    for(i = 0; i < id; ++i) {
      if(pBi >= randkernel(state, kNeuron)) {
        dev_conVec[id + i * N_NEURONS] = 1; // id --> i
        dev_conVec[i + id * N_NEURONS] = 1; //  i --> id
      }
      else {
        if(2 * pUni >= randkernel(state, kNeuron)) {
          if(randkernel(state, kNeuron) > 0.5) {
            dev_conVec[id + i * N_NEURONS] = 1; // id --> i
          }
          else {
            dev_conVec[i + id * N_NEURONS] = 1; //  i --> id
          }
        }
      }
    }
  }
}


void IsSquare(unsigned long long x, unsigned long long y) {
  double z, IF_EXIT = 0;
  z = sqrt(x);
  if((unsigned long long)z * z != x) {
    IF_EXIT = 1;
    printf("\n NE is not a perfect square ! \n");
    printf("next perfect square is : %llu \n", (unsigned long long)(ceil(z) * ceil(z)));
  }
  z = sqrt(y);
  if((unsigned long long)z * z != y) {
    IF_EXIT = 1;
    printf("\n NI is not a perfect square ! \n");
    printf("next perfect square is : %llu \n", (unsigned long long)(ceil(z) * ceil(z)));
  }
  if(IF_EXIT) {
    printf("\n\n Connection Matrix not generated !!\n");
    exit(-1);
  }
}

int main(int argc, char *argv[]) {
  int i, nChunks = 1, deviceId = 0, maxNeurons = N_NEURONS, bidirType = 0;
  float *dev_conVecPtr, *conVec;
  /*  int fullConVecE[NE * NE], fullConVecI[NI *NI], fullConvecIE[NE*NI], fullConVecEI[NI*NE];*/
  float*fullConVec = NULL, *conProbMat = NULL;
  FILE *fpConVec;
  cudaDeviceProp prop;
  unsigned long maxMem = 12079136768;
  enum ConMat_type {
    random, sparseE2E, distDependent, biDir, fixedEII, random_with_diffK
  };
  ConMat_type conMatType = biDir; 
  if(argc > 1) {
    if(atoi(argv[1]) == 0) 
      conMatType = random;
    if(atoi(argv[1]) == 1)
      conMatType = distDependent; 
    if(atoi(argv[1]) == 2)
      conMatType = sparseE2E;
    if(atoi(argv[1]) == 3)
      conMatType = biDir;/* DEFAULT */
    if(atoi(argv[1]) == 5)
      conMatType = fixedEII;
    if(atoi(argv[1]) == 6)
      conMatType = random_with_diffK;
  }
  if(argc >2) {
    //    if(atoi(argv[2]) == 1) {
    bidirType = atoi(argv[2]);
      //}
  }
  cudaCheck(cudaGetDeviceProperties(&prop, deviceId));
  printf("Global Mem = %ld\n", prop.totalGlobalMem);
  i = 0;
  maxMem = prop.totalGlobalMem;
  if(maxMem < (N_NEURONS * N_NEURONS * 4 + N_NEURONS * 4)) {
    while(maxMem < ((N_NEURONS / nChunks) * N_NEURONS * 4   + N_NEURONS * 5)) {
      nChunks += 1;
    }
    maxNeurons = N_NEURONS / nChunks;
  }
  /*  if(maxNeurons > 30000) { nChunks += 2;}*/
  maxNeurons = N_NEURONS / nChunks;
  printf(" maxNeurons = %d\n nChunks = %d\n", maxNeurons, nChunks);
  curandState *devStates, *fixedStates;
  fullConVec = (float *)malloc((unsigned long long)N_NEURONS * N_NEURONS * sizeof(float));
  conProbMat = (float *)malloc((unsigned long long)N_NEURONS * N_NEURONS * sizeof(float));
  if(fullConVec == NULL) {
    printf("fullconvec not assigned\n"); 
    exit(-1);
  }
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 512;
  int BlocksPerGrid = (N_NEURONS + ThreadsPerBlock - 1) / ThreadsPerBlock;
  if(BlocksPerGrid > 65536) {
    printf("BlocksPerGrid exceds valid number of allowed blocks of 65536");
    exit(-1);
  }
  fpConVec = fopen("conVec.dat", "wb"); 
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  cudaCheck(cudaMallocHost((void **)&conVec, (N_NEURONS / nChunks) * N_NEURONS * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(float)));
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  cudaCheckLastError("setup_kernel failed\n");
  unsigned long long int chunckSize = ((unsigned long long)N_NEURONS / nChunks) * N_NEURONS;
  printf("chunckSize = %llu \n ", chunckSize);
  BlocksPerGrid = (maxNeurons + ThreadsPerBlock - 1) / ThreadsPerBlock;
  printf("Threads per block : %d, Blocks per grid : %d \n", ThreadsPerBlock, BlocksPerGrid);
  for(unsigned long long int i = 0; i < nChunks; ++i) {
    printf("generating chunk %llu ... ", i);fflush(stdout);
    initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, maxNeurons);
    switch(conMatType) {
    case random:
      printf("\n random conmat \n");
      kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons);
      break;
    case random_with_diffK:
      printf("\n random conmat with different K's for each population \n");
      kernelGenConMatWithDiffK<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons);
      break;
    case distDependent:
      printf("\n anatomic conmat \n");
      /* ARRANGE NEURONS ON A SQUARE GRID, REQUIRES THAT SQRT(NA) IS AN INTEGER */
      IsSquare(NE, NI);
      KernelGenConProbMat<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr);
      KernelConProbPreFactor<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr);
      cudaCheck(cudaMemcpy(conProbMat, dev_conVecPtr, (unsigned long long)N_NEURONS * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
      KernelGenDistDepConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons);
      break;
    case sparseE2E:
      kernelGenConMatSparseE2E<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons);
      break;
    case biDir:
      if(i == 0){
	printf("\nBi-dir\n");
	ConMatBiDir(fullConVec, bidirType); // defined in file : tinyRNG.cu
	TestBidir(fullConVec);
	}
       break;
    case fixedEII:
      cudaCheck(cudaMalloc((void **)&fixedStates,  N_NEURONS * sizeof(curandState)));
      setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(fixedStates, 1326783ULL);
      printf("fixed EI, IE, II\n");
      ConMatFixedEII(fullConVec);
      /*      kernelFixedEII<<<BlocksPerGrid, ThreadsPerBlock>>>(fixedStates, devStates, dev_conVecPtr, i, maxNeurons);*/
    default:
      kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons);
    }

    if(conMatType != biDir) {
      printf("done\ncopying dev to host ...");
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
      printf(" done\n");
      printf("#Elements in a chunk: %llu \n", (N_NEURONS / nChunks) * N_NEURONS);
      printf("#Elements in a chunk: %llu \n", chunckSize);            

      // for(int ii = 0; ii < chunckSize; ++ii) {
      // 	conVec[ii]  = ii + i * chunckSize;
      // }
      // printf("------------------------\n");	
      // for(int ii = 0; ii < chunckSize; ++ii) {

      // 	printf("%d ", (int)conVec[ii]);
      // }
      // printf("\n");	
      // printf("------------------\n");	
      // for(int ii = 0; ii < 4; ++ii) {
      // 	  printf("%d %d\n", (int)conVec[ii], (int)conVec[ii + N_NEURONS]);
      // 	}

      // }	
      // else {
      // 	for(int ii = 0; ii < 4; ++ii) {
      // 	  //	  printf("%d %d\n", (int)conVec[ii + i * chunckSize], (int)conVec[ii + N_NEURONS + i * chunckSize]);
      // 	  printf("%d %d\n", (int)(ii + i * chunckSize), (int)(ii + N_NEURONS + i * chunckSize));	  
      // 	}
      // }
      
      for(unsigned long long int j = 0; j < chunckSize; ++j) {
	/*      printf("%du\n", j + chunckSize * i);*/
        fullConVec[j + chunckSize * i] = conVec[j];
      } 
    }
  }

	  // for(int ii = 0; ii < 4; ++ii) {
	  //   printf("%d %d %d %d\n", (int)fullConVec[ii], (int)fullConVec[ii + N_NEURONS], (int)fullConVec[ii + chunckSize],  (int)fullConVec[ii + N_NEURONS + chunckSize]);
	  // }

  
  //   printf("done\ncopying dev to host ...");
  //   cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
  //   printf(" done\n");
  //   for(unsigned long long int j = 0; j < chunckSize; ++j) {
  //     /*      printf("%du\n", j + chunckSize * i);*/
  //     fullConVec[j + chunckSize * i] = conVec[j];
  //   } 
  // }


  fclose(fpConVec);
  cudaFreeHost(conVec);
  int idxVec[N_NEURONS], nPostNeurons[N_NEURONS];
  int *sparseConVec;
  sparseConVec = (int *)malloc((unsigned long long)N_NEURONS * (2ULL + (unsigned long long)K + N_NEURONS) * sizeof(int));
  printf("generating sparse representation ..."); fflush(stdout);
  GenSparseMat(fullConVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  fpSparseConVec = fopen("sparseConVec.dat", "wb");
  unsigned long int nElementsWritten, nConnections = 0;
  for(i = 0; i < N_NEURONS; ++i) {
    nConnections += nPostNeurons[i];
  }
  printf("done\n#connections = %lu\n", nConnections);
  printf("writing to file ... "); fflush(stdout);
  fpSparseConVec = fopen("sparseConVec.dat", "wb");
  //  fwrite(sparseConVec, sizeof(*sparseConVec), N_NEURONS * (2 * (int)K + N_NEURONS), fpSparseConVec);
  nElementsWritten = fwrite(sparseConVec, sizeof(*sparseConVec), nConnections, fpSparseConVec);
  fclose(fpSparseConVec);
  fpIdxVec = fopen("idxVec.dat", "wb");
  fwrite(idxVec, sizeof(*idxVec), N_NEURONS,  fpIdxVec);
  fclose(fpIdxVec);
  fpNpostNeurons = fopen("nPostNeurons.dat", "wb");
  fwrite(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpNpostNeurons);
  printf("done\n");
  /*
  fpSparseConVec = fopen("sparseConVec.dat", "rb");
  fpIdxVec = fopen("idxVec.dat", "rb");
  fpNpostNeurons = fopen("nPostNeurons.dat", "rb");
  fread(sparseConVec, sizeof(*sparseConVec), N_NEURONS * (2 * K + 1), fpSparseConVec);
  fread(idxVec, sizeof(*idxVec), N_NEURONS, fpIdxVec);
  fread(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpSparseConVec);
  fclose(fpIdxVec);
  fclose(fpNpostNeurons);*/
 
 if(N_NEURONS < 2) {
    for(i = 0; i < N_NEURONS; ++i) {
      printf("neuron %d projects to : ", i);
      for(int j = 0; j < nPostNeurons[i]; ++j) {
        printf("%d ", sparseConVec[idxVec[i] + j]);
      }
      printf("\n");
    }
  }

  /*
  int buffer[N_NEURONS * N_NEURONS], j;
  fp = fopen("conVec.dat", "rb");
  fread(buffer, sizeof(int), N_NEURONS * N_NEURONS, fp);
  fclose(fp);*/
  
//  printf("\n !!!!! convec.csv ..."); fflush(stdout);
  FILE *fp, *fp01, *fpConMat;
  /*  int nEE[NE], nEI[NE], nIE[NI], nII[NI];*/
  /*  int ncounts[N_NEURONS];*/
  fpConMat = fopen("conMat.csv", "w");
  fp01 = fopen("countI.csv", "w");  fp = fopen("countE.csv", "w");
  printf("\nN = %llu\n", N_NEURONS);
  int countE = 0, countI = 0;
  for(i = 0; i < N_NEURONS; ++i) {
    // here index i is clm and j is row 
    countI = 0;
    countE = 0;
    for(int j = 0; j < N_NEURONS; ++j) {
      /*      fprintf(fpConMat, "%1.1f ", fullConVec[i + j * N_NEURONS]);*/
      //      fprintf(fpConMat, "%1.1f ", conProbMat[i + j * N_NEURONS]);
      //      printf("%1.1f ", conProbMat[i + j * N_NEURONS]);
      //      printf("\n %d \n", (int)(i * N_NEURONS + j));
      //      fprintf(stdout, "%d ", (int)fullConVec[i + j * N_NEURONS]);
      if(j < NE) {
        countE += fullConVec[i * N_NEURONS + j];   
      }
      else {
        countI += fullConVec[i * N_NEURONS + j];   
      }
    }
    fprintf(fp, "%d\n", countE); 
    fprintf(fp01, "%d\n", countI);
    //    fprintf(stdout, "\n");
    //    fprintf(fpConMat, "\n");
  }
  fprintf(stdout, " done\n");
  free(conProbMat);
  fclose(fp);   
  fclose(fp01);
  fclose(fpConMat);
  free(fullConVec);
  free(sparseConVec);
  return 0;
}


