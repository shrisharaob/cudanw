#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mycurand.h"
#include <stdio.h>
#include "devHostConstants.h"
#include "GenSparseMat.cu"
#include "GenConProbDistDepMat.cu"

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
    /*    if(id < N_NEURONS & NE > 0) {*/
    if(kNeuron < NE) {
      n = (float)NE;
    }
    else {
      n = (float)NI;
    }
    for(i = 0; i < N_NEURONS; ++i) {
      if(i < NE) {  /* E --> E */
        if(k/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVec[id + i * maxNeurons] = 1;
        }
      }
      if(i >= NE) { /* E --> I */
        if(k/n >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVec[id + i * maxNeurons] = 1;
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
  int i, nChunks = 1, deviceId = 0, maxNeurons = N_NEURONS;
  float *dev_conVecPtr, *conVec;
  /*  int fullConVecE[NE * NE], fullConVecI[NI *NI], fullConvecIE[NE*NI], fullConVecEI[NI*NE];*/
  float*fullConVec = NULL, *conProbMat = NULL;
  FILE *fpConVec;
  cudaDeviceProp prop;
  unsigned long maxMem = 12079136768;
  enum ConMat_type {
    random, sparseE2E, distDependent 
  };
  ConMat_type conMatType = distDependent;
  if(argc > 1) {
    if(atoi(argv[1]) == 0) 
      conMatType = random;
    if(atoi(argv[1]) == 1)
      conMatType = distDependent; /* DEFAULT */
    if(atoi(argv[1]) == 2)
      conMatType = sparseE2E;
  }
      
  cudaCheck(cudaGetDeviceProperties(&prop, deviceId));
  printf("Global Mem = %ld\n", prop.totalGlobalMem);
  i = 0;
  maxMem = prop.totalGlobalMem;
  if(maxMem < (N_NEURONS * N_NEURONS * 4 + N_NEURONS * 5)) {
    while(maxMem < ((N_NEURONS / nChunks) * N_NEURONS * 4.0   + N_NEURONS * 5.0)) {
      nChunks += 1;
    }
    maxNeurons = N_NEURONS / nChunks;
  }
  /*  if(maxNeurons > 30000) { nChunks += 2;}*/
  maxNeurons = N_NEURONS / nChunks;
  printf(" maxNeurons = %d\n nChunks = %d\n", maxNeurons, nChunks);
  curandState *devStates;
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
      kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons);
      break;
    case distDependent:
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
    default:
      kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons);
    }
    printf("done\ncopying dev to host ...");
    cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
    printf(" done\n");
    for(unsigned long long int j = 0; j < chunckSize; ++j) {
      /*      printf("%du\n", j + chunckSize * i);*/
      fullConVec[j + chunckSize * i] = conVec[j];
    } 
  }
  fclose(fpConVec);
  cudaFreeHost(conVec);
  int idxVec[N_NEURONS], nPostNeurons[N_NEURONS];
  int *sparseConVec;
  sparseConVec = (int *)malloc((unsigned long long)N_NEURONS * (2ULL + (unsigned long long)K + N_NEURONS) * sizeof(int));
  printf("generating sparse representation ..."); fflush(stdout);
  GenSparseMat(fullConVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);

  printf("done\n writing to file ... "); fflush(stdout);
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  fpSparseConVec = fopen("sparseConVec.dat", "wb");
  fwrite(sparseConVec, sizeof(*sparseConVec), N_NEURONS * (2 * (int)K + N_NEURONS), fpSparseConVec);
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
  if(N_NEURONS < 20) {
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
  
  printf("convec.csv ..."); fflush(stdout);
  FILE *fp, *fp01, *fpConMat;
  /*  int nEE[NE], nEI[NE], nIE[NI], nII[NI];*/
  /*  int ncounts[N_NEURONS];*/
  fpConMat = fopen("conMat.csv", "w");
  fp01 = fopen("countI.csv", "w");  fp = fopen("countE.csv", "w");
  printf("\nN = %llu\n", N_NEURONS);
  int countE = 0, countI = 0;
  for(i = 0; i < N_NEURONS; ++i) {
    countI = 0;
    countE = 0;
    for(int j = 0; j < N_NEURONS; ++j) {
      fprintf(fpConMat, "%1.1f ", fullConVec[i + j * N_NEURONS]);
      /*      fprintf(fpConMat, "%1.1f ", conProbMat[i + j * N_NEURONS]);*/
      /*	fprintf(stdout, "%d ", fullConVec[i + j * N_NEURONS]);*/
      if(j < NE) {
        countE += fullConVec[i * N_NEURONS + j];   
      }
      else {
        countI += fullConVec[i * N_NEURONS + j];   
      }
    }
    fprintf(fp, "%d\n", countE); 
    fprintf(fp01, "%d\n", countI);
    fprintf(fpConMat, "\n");
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


