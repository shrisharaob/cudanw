#ifndef _CUDARANDFUNCS_
#define _CUDARANDFUNCS_
#include <cuda.h>
//#include "curand_kernel.h"
#include "mycurand.h"
#include "devFunctionProtos.h"
#include "devHostConstants.h"

__global__ void setup_kernel(curandState *state, unsigned long long seed ) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    if(id < N_NEURONS) {
      curand_init(seed * (id + 7), id, 0, &state[id]);
    }
}

__device__ float randkernel(curandState *state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  float randNumber;
  curandState localState = state[id]; // state in global memory 
  randNumber = curand_uniform(&localState);
  state[id] = localState;
  return randNumber;
}


__global__ void kernelGenConMat(curandState *state, int *dev_conVec){
  /* indexing of matrix row + clm x N_NEURONS*/
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int i;
  float k, n;
  
  if(id < N_NEURONS) {
    k = (float)K;
    /* E --> EI */
    if(id < NE & NE > 0) {
      n = (float)NE;
      for(i = 0; i < N_NEURONS; ++i) {
        if(i < NE) {  /* E --> E */
          if(k/n >= randkernel(state)) { /* neuron[id] receives input from i ? */
            dev_conVec[id + i * N_NEURONS] = 1;
          } 
        }
        if(i > NE) { /* E --> I */
          if(k/n >= randkernel(state)) { /* neuron[id] receives input from i ? */
            dev_conVec[id + i * N_NEURONS] = 1;
          } 
        }
      }
    }

    /* I --> EI */
    if(id > NE & NI > 0) {
      n = (float)NI;
      for(i = 0; i < N_NEURONS; ++i) {
        if(i < NE) {  /* I --> E */
          if(k/n >= randkernel(state)) { /* neuron[id] receives input from i ? */
            dev_conVec[id + i * N_NEURONS] = 4;
          } 
        }
        if(i > NE) { /* I --> I */
          if(k/n >= randkernel(state)) { /* neuron[id] receives input from i ? */
            dev_conVec[id + i * N_NEURONS] = 1;
          } 
        }
      }
    }
  }
}
#endif
