#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mycurand.h"
#include <stdio.h>
#define NE 10
#define NI 10
#define N_NEURONS (NE + NI)
#define K 2.0

void __cudaCheck(cudaError err, const char* file, const int line);
#define cudaCheck(err) __cudaCheck (err, __FILE__, __LINE__)
void __cudaCheckLastError(const char* errorMessage, const char* file, const int line);
#define cudaCheckLastError(msg) __cudaCheckLastError (msg, __FILE__, __LINE__)

void __cudaCheck(cudaError err, const char *file, const int line) {
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
      file, line, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

void __cudaCheckLastError(const char *errorMessage, const char *file, const int line) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
      file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

__global__ void initConVec(int *dev_conVec, int maxNeurons) {
  int mNeuron = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  int i;
  if(mNeuron < maxNeurons) {
    for(i = 0; i < N_NEURONS; ++i) {
      dev_conVec[mNeuron + maxNeurons * i] = 0;
    }
    /*  mNeuron += stride;*/
  }
}

__global__ void setup_kernel(curandState *state, unsigned long long seed ) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    if(id < N_NEURONS) {
      curand_init(seed * (id + 7), id, 0, &state[id]);
    }
}

__device__ float randkernel(curandState *state, int lChunck) {
  /*RETURNS ONE SAMPLE FROM UNIFORM DISTRIBUTION*/
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  float randNumber;
  curandState localState = state[id]; /* state in global memory */
  randNumber = curand_uniform(&localState);
  state[id] = localState;
  return randNumber;
}

__global__ void kernelGenConMat(curandState *state, int *dev_conVec, int lChunck, int maxNeurons){
  /* indexing of matrix row + clm x N_NEURONS*/
  int id = threadIdx.x + blockIdx.x * blockDim.x;
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
  FILE *fpConVec;
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceId));
  printf("Global Mem = %ld\n", prop.totalGlobalMem);
  if(prop.totalGlobalMem < (N_NEURONS * N_NEURONS * 4 + N_NEURONS * 5)) {
    while(prop.totalGlobalMem < ((N_NEURONS / nChunks) * N_NEURONS * 4.0   + N_NEURONS * 5.0)) {
      nChunks += 1;
    }
    maxNeurons = N_NEURONS / nChunks;
  }

  curandState *devStates;
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 512;
  int BlocksPerGrid = (N_NEURONS + ThreadsPerBlock - 1) / ThreadsPerBlock;
  if(BlocksPerGrid > 65536) {
    printf("BlocksPerGrid exceds valid number of allowed blocks of 65536");
    exit(-1);
  }
  printf("Threads per block : %d, Blocks per grid : %d \n", ThreadsPerBlock, BlocksPerGrid);
  fpConVec = fopen("conVec.dat", "wb"); nChunks = 2; maxNeurons = N_NEURONS / nChunks;
  printf(" maxNeurons = %d\n nChunks = %d\n", maxNeurons, nChunks);
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  cudaCheck(cudaMallocHost((void **)&conVec, (N_NEURONS / nChunks) * N_NEURONS * sizeof(*conVec)));
  cudaCheck(cudaMalloc((void **)&dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(*conVec)));
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  for(i = 0; i < nChunks; ++i) {
    printf("chunk %d\n", i);
    initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, maxNeurons);
    kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, nChunks, maxNeurons);
    printf("done\n");
    cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (N_NEURONS / nChunks) * N_NEURONS * sizeof(int), cudaMemcpyDeviceToHost));
    printf("cpy done\n");
    /* WRITE TO BINARY FILE */
    fwrite(conVec, (N_NEURONS / nChunks) * N_NEURONS, sizeof(*conVec), fpConVec);
  }
  fclose(fpConVec);
  cudaFreeHost(conVec);
  /*FILE *fp;
  int buffer[N_NEURONS * N_NEURONS], j;
  fp = fopen("conVec.dat", "rb");
  fread(buffer, sizeof(int), N_NEURONS * N_NEURONS, fp);
  fclose(fp);
  fp = fopen("conMat.csv", "w");
  printf("\nN = %d\n", N_NEURONS);
  for(i = 0; i < N_NEURONS; ++i) {
    for(j = 0; j < N_NEURONS; ++j) {
      fprintf(fp, "%d ", buffer[i + j * N_NEURONS]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);*/
  printf("\n");  
  return 0;
}
