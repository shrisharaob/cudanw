#ifndef _DEV_FUNC_PROTOS_
#define _DEV_FUNC_PROTOS_
#include "cuda.h"
#include "mycurand.h"


__device__ void rk4(double *y, double *dydx, int n, double rk4X, double h, double *yout, double iSynap, double ibg, double iff);

__device__ void derivs(double t, double stateVar[], double dydx[], double isynap, double ibg, double iff);

__device__ double isynap(double vm, int *dev_conVec);

__device__ double bgCur(double vm);

__device__ void Gff(double t);

__device__ void RffTotal(double t);

__device__ double IFF(double vm);

__device__ double IFF_orimap(double vm);

__device__ double normRndKernel(curandState *state);

__device__ double randkernel(curandState *state);

__device__ double atomicAdd(double* address, double val);

__device__ double OrientationAngleForGivenNeuron(unsigned int neuronId);
__device__ int OrientationForGivenNeuron(unsigned int neuronId);
__device__ double MyDivide(double x, double y);
__device__ double XCordinate(unsigned long int neuronIdx);
__device__ double YCordinate(unsigned long int neuronIdx);
/* ======================= GLOBAL KERNELS ============================================== */

__global__ void rkdumbPretty(kernelParams_t, devPtr_t);

__global__ void AuxRffTotal(curandState *, curandState *);

/*
__global__ void rkdumb(double vstart[], int nvar, double x1, double x2, 
//                       int nstep, int *nSpks, double *spkTimes, int *spkNeuronId, double *y, int *dev_conVec, double *, double *);

//__global__ void setup_kernel(curandState *state, unsigned long long seed );


//__global__ void kernelGenConMat(curandState *state, int nNeurons, int *dev_conVec);
*/

#endif
