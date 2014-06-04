#ifndef _DEV_FUNC_PROTOS_
#define _DEV_FUNC_PROTOS_
#include "cuda.h"
#include "mycurand.h"


__device__ void rk4(float *y, float *dydx, int n, float rk4X, float h, float *yout, float iSynap);

__device__ void derivs(float t, float stateVar[], float dydx[], float isynap);

__device__ float isynap(float vm, int *dev_conVec);

//__device__ float randkernel(curandState *state);


// ======================= GLOBAL KERNELS ============================================== \\

__global__ void rkdumb(float x1, float x2, int nstep, int *nSpks, float *spkTimes, 
                       int *spkNeuronId, float *y, int *dev_conVec, float *isynap, curandState *dev_state);

//__global__ void rkdumb(float vstart[], int nvar, float x1, float x2, 
//                       int nstep, int *nSpks, float *spkTimes, int *spkNeuronId, float *y, int *dev_conVec, float *, float *);

//__global__ void setup_kernel(curandState *state, unsigned long long seed );


//__global__ void kernelGenConMat(curandState *state, int nNeurons, int *dev_conVec);


#endif
