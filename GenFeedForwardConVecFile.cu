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

void MatTranspose(float *m, int w, int h) {
    // transpose of flattened matrix 
      // m : flattened matrix  -  row + clm * linearId
      // w : # of columns
      // h : # of rows
      int start, next, i;
    double tmp;
    for (start = 0; start <= w * h - 1; start++) {
        next = start;
        i = 0;
        do {i++;
            next = (next % h) * w + next / h;
          } while (next > start);
        if (next < start || i == 1) continue;
        tmp = m[next = start];
        do {
            i = (next % h) * w + next / h;
            m[next] = (i == start) ? tmp : m[i];
            next = i;
          } while (next > start);
      }
  }

__global__ void KernelGenFeedForwardDistDepConMat(curandState *state, float *dev_conVecFF, int lChunck, int maxNeurons){
  /* GENERATE FEED FORWARD CONNECTION MATRIX WITH ANOTOMIC CONNECTIVITY PROFILE */
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  if(id < maxNeurons & kNeuron < N_NEURONS) {
    for(i = 0; i < NFF; ++i) {
        if(dev_conVecFF[id + i * maxNeurons] >= randkernel(state, kNeuron)) { /* neuron[id] receives input from i ? */
          dev_conVecFF[id + i * maxNeurons] = 1;
        }
        else{
          dev_conVecFF[id + i * maxNeurons] = 0;
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
  cudaDeviceProp prop;
  unsigned long maxMem = 12079136768;
  int IF_FF_ORI_MAP = 1;
  enum ConMat_type {
    random, sparseE2E, distDependent, biDir, fixedEII
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
  if(maxMem < (N_NEURONS * N_NEURONS * 4 + N_NEURONS * 5)) {
    while(maxMem < ((N_NEURONS / nChunks) * N_NEURONS * 4.0   + N_NEURONS * 5.0)) {
      nChunks += 1;
    }
    maxNeurons = N_NEURONS / nChunks;
  }
  /*  if(maxNeurons > 30000) { nChunks += 2;}*/
  maxNeurons = N_NEURONS / nChunks;
  printf(" maxNeurons = %d\n nChunks = %d\n", maxNeurons, nChunks);
  curandState *devStates, *fixedStates;
  
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = 512;
  int BlocksPerGrid = (N_NEURONS + ThreadsPerBlock - 1) / ThreadsPerBlock;
  if(BlocksPerGrid > 65536) {
    printf("BlocksPerGrid exceds valid number of allowed blocks of 65536");
    exit(-1);
  }
      printf("\n generating feed forward conmat \n");
      printf(" # Feed forward neurons = %f", CFF*K);
      /* ARRANGE NEURONS ON A SQUARE GRID, REQUIRES THAT SQRT(CFF * K) IS AN INTEGER */
      unsigned int kff, sqrtNff;
      kff = (unsigned int)CFF * K;
      //      sqrtKff = (unsigned int)sqrt(kff);
      sqrtNff = (unsigned int)sqrt(NFF);
      if(sqrtNff * sqrtNff != NFF) {
        printf("the number of feede forward neurons (KFF) cannot be placed on square !!!");
        exit(-1);
      }
      
      curandState *devStatesFF;
      float *dev_conVecPtrFF, *conVecFF, *conProbMatFF;
      conProbMatFF = (float *)malloc((unsigned long long)NFF * N_NEURONS * sizeof(float));
      cudaCheck(cudaMalloc((void **)&devStatesFF,  N_NEURONS * sizeof(curandState)));
      cudaCheck(cudaMallocHost((void **)&conVecFF, (unsigned long long)NFF * N_NEURONS * sizeof(float)));
      cudaCheck(cudaMalloc((void **)&dev_conVecPtrFF, (unsigned long long)NFF * N_NEURONS * sizeof(float)));
      setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStatesFF, time(NULL));
      KernelGenFeedForwardConProbMat<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtrFF, 0); // second arg is IF_PERIODIC

      cudaCheck(cudaMemcpy(conProbMatFF, dev_conVecPtrFF, (unsigned long long)NFF * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));


      MatTranspose(conProbMatFF, NFF, N_NEURONS);
      cudaCheck(cudaMemcpy(dev_conVecPtrFF, conProbMatFF, (unsigned long long)NFF * N_NEURONS * sizeof(float), cudaMemcpyHostToDevice));


      KernelFeedForwardConProbPreFactor<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtrFF);


      //      cudaCheck(cudaMemcpy(conProbMatFF, dev_conVecPtrFF, (unsigned long long)NFF * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
      
      KernelGenFeedForwardDistDepConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStatesFF, dev_conVecPtrFF, i, maxNeurons);
      cudaCheck(cudaMemcpy(conVecFF, dev_conVecPtrFF, (unsigned long long)NFF * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));

      // printf("\n");
      // for(i = 0; i < N_NEURONS; ++i) {
      //   for(int j = 0; j < NFF; ++j) {
      //     printf("%d ", (int)conVecFF[j + i * NFF]);
      //   }
      //   printf("\n");
      // }
      /* GENERATE SPARSE REPRESENTATIONS */
      int idxVecFF[NFF], nPostNeuronsFF[NFF];
      int *sparseConVec;
      sparseConVec = (int *)malloc((unsigned long long)N_NEURONS * (2ULL * (unsigned long long)kff + NFF) * sizeof(int)); 
      printf("generating sparse representation ..."); fflush(stdout);
      GenSparseFeedForwardMat(conVecFF, NFF, N_NEURONS, sparseConVec, idxVecFF, nPostNeuronsFF);
      printf("done\n writing to file ... "); fflush(stdout);
      FILE *fpSparseConVecFF, *fpIdxVecFF, *fpNpostNeuronsFF;
      fpSparseConVecFF = fopen("sparseConVecFF.dat", "wb");
      unsigned int nElementsWritten;
      nElementsWritten = fwrite(sparseConVec, sizeof(*sparseConVec), N_NEURONS * (2 * (int)kff + NFF), fpSparseConVecFF);
      printf("\sparseconvec: #n= %d\n", nElementsWritten);
      fclose(fpSparseConVecFF);

      fpIdxVecFF = fopen("idxVecFF.dat", "wb");
      fwrite(idxVecFF, sizeof(*idxVecFF), N_NEURONS,  fpIdxVecFF);
      fclose(fpIdxVecFF);
      fpNpostNeuronsFF = fopen("nPostNeuronsFF.dat", "wb");
      fwrite(nPostNeuronsFF, sizeof(*nPostNeuronsFF), N_NEURONS, fpNpostNeuronsFF);
      fclose(fpNpostNeuronsFF);
      printf("done\n");
      int countFF= 0;
      FILE *FFfp0;
      FFfp0 = fopen("ffcount.csv", "w");

      FILE *fpconmat = fopen("ffcm.csv", "w");
      for(i = 0; i < N_NEURONS; ++i) {
        for(int j = 0; j < NFF; ++j) {
          fprintf(fpconmat, "%f ", conProbMatFF[i * NFF + j  ]); 
        }
        fprintf(fpconmat, "\n");
      }
      fclose(fpconmat);

      free(conProbMatFF);


      if(N_NEURONS < 2) {
        for(i = 0; i < NFF; ++i) {
          printf("neuron %d projects to : ", i);
          //          printf("%d", nPostNeuronsFF[i]);
          for(int j = 0; j < nPostNeuronsFF[i]; ++j) {
            printf("%d ", sparseConVec[idxVecFF[i] + j]);
          }
          printf("\n");
        }
      }



      for(i = 0; i < N_NEURONS; ++i) {
        countFF = 0;
        for(int j = 0; j < NFF; ++j) {
      //printf("%d ", (int)conVecFF[i + j * N_NEURONS]);
          countFF += (int)conVecFF[i * NFF + j];   
        }
        fprintf(FFfp0, "%d\n", countFF); 
      }
      fclose(FFfp0);
      

  return 0;
}


