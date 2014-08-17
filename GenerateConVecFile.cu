#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mycurand.h"
#include <stdio.h>
#include "devHostConstants.h"
#include "GenSparseMat.cu"


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

__global__ void initConVec(int *dev_conVec, int maxNeurons) {
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

__device__ float randkernel(curandState *state, int lChunck) {
  /*RETURNS ONE SAMPLE FROM UNIFORM DISTRIBUTION*/
  unsigned int id = (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  float randNumber;
  curandState localState = state[id]; /* state in global memory */
  randNumber = curand_uniform(&localState);
  state[id] = localState;
  return randNumber;
}

__global__ void kernelGenConMat(curandState *state, int *dev_conVec, int lChunck, int maxNeurons){
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  int i;
  float k, n;
  if(id < maxNeurons) {
    k = (float)K;
    /* E --> EI */
    if(id < NE & NE > 0) {
      n = (float)NE;
      for(i = 0; i < N_NEURONS; ++i) {
        if(i < NE) {  /* E --> E */
          if(k/n >= randkernel(state, lChunck)) { /* neuron[id] receives input from i ? */
            dev_conVec[id + i * maxNeurons] = 1;
          }
        }
        if(i >= NE) { /* E --> I */
          if(k/n >= randkernel(state, lChunck)) { /* neuron[id] receives input from i ? */
            dev_conVec[id + i * maxNeurons] = 1;
          } 
        }
      }
    }
    /* I --> EI */
    if(id >= NE & NI > 0) {
      n = (float)NI;
      for(i = 0; i < N_NEURONS; ++i) {
        if(i < NE) {  /* I --> E */
          if(k/n >= randkernel(state, lChunck)) { /* neuron[id] receives input from i ? */
            dev_conVec[id + i * maxNeurons] = 1;
          } 
        }
        if(i >= NE) { /* I --> I */
          if(k/n >= randkernel(state, lChunck)) { /* neuron[id] receives input from i ? */
            dev_conVec[id + i * maxNeurons] = 1;
          } 
        }
      }
    }
  }
}

int main() {
  int *conVec, *dev_conVecPtr, i, nChunks = 1, deviceId = 0, maxNeurons = N_NEURONS;
  /*  int fullConVecE[NE * NE], fullConVecI[NI *NI], fullConvecIE[NE*NI], fullConVecEI[NI*NE];*/
  int *fullConVec = NULL;
  FILE *fpConVec;
  cudaDeviceProp prop;
  unsigned long maxMem = 12079136768;
  /*  cudaCheck(cudaGetDeviceProperties(&prop, deviceId));*/
  /*  printf("Global Mem = %ld\n", prop.totalGlobalMem);*/
  i = 0;
  if(maxMem < (N_NEURONS * N_NEURONS * 4 + N_NEURONS * 5)) {
    while(maxMem < ((N_NEURONS / nChunks) * N_NEURONS * 4.0   + N_NEURONS * 5.0)) {
      nChunks += 1;
    }
    maxNeurons = N_NEURONS / nChunks;
  }

  curandState *devStates;
  fullConVec = (int *)malloc((unsigned long long)N_NEURONS * N_NEURONS * sizeof(int));

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
  printf("Threads per block : %d, Blocks per grid : %d \n", ThreadsPerBlock, BlocksPerGrid);
  fpConVec = fopen("conVec.dat", "wb"); 
  if(maxNeurons > 30000) { nChunks += 2;}
  maxNeurons = N_NEURONS / nChunks;
  printf(" maxNeurons = %d\n nChunks = %d\n", maxNeurons, nChunks);
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  cudaCheck(cudaMallocHost((void **)&conVec, (N_NEURONS / nChunks) * N_NEURONS * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(int)));
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  cudaCheckLastError("setup_kernel failed\n");
  unsigned long long int chunckSize = ((unsigned long long)N_NEURONS / nChunks) * N_NEURONS;
  printf("chunckSize = %lu \n ", chunckSize);
  for(unsigned long long int i = 0; i < nChunks; ++i) {
    printf("generating chunk %d ... ", i);fflush(stdout);
    initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, maxNeurons);
    kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, nChunks, maxNeurons);
    printf("done\ncopying dev to host ...");
    cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(int), cudaMemcpyDeviceToHost));
    printf(" done\n");
    for(unsigned long long int j = 0; j < chunckSize; ++j) {
      /*      printf("%du\n", j + chunckSize * i);*/
      fullConVec[j + chunckSize * i] = conVec[j];
    } 
  }
  fclose(fpConVec);
  cudaFreeHost(conVec);
  int sparseConVec[N_NEURONS * (2 * (int)K + N_NEURONS)], idxVec[N_NEURONS], nPostNeurons[N_NEURONS];
  printf("generating sparse representation ..."); fflush(stdout);
  GenSparseMat(fullConVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);
  free(fullConVec);
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
  if(N_NEURONS < 10) {
    for(i = 0; i < N_NEURONS; ++i) {
      printf("neuron %d projects to : ", i);
      for(int j = 0; j < nPostNeurons[i]; ++j) {
	printf("%d ", sparseConVec[idxVec[i] + j]);
      }
      printf("\n");
    }
  }
  /*FILE *fp;
  int buffer[N_NEURONS * N_NEURONS], j;
  fp = fopen("conVec.dat", "rb");
  fread(buffer, sizeof(int), N_NEURONS * N_NEURONS, fp);
  fclose(fp);
  fp = fopen("conMat.csv", "w");
  printf("\nN = %d\n", N_NEURONS);
  for(i = 0; i < N_NEURONS; ++i) {
    for(int j = 0; j < N_NEURONS; ++j) {
      fprintf(fp, "%d ", fullConVec[i + j * N_NEURONS]);
      fprintf(stdout, "%d ", fullConVec[i + j * N_NEURONS]);
    }
    fprintf(fp, "\n");
    fprintf(stdout, "\n");
  }
  fclose(fp);*/
  return 0;
}
