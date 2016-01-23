#ifndef _IFFCURRENT_
#define _IFFCURRENT_
#include "globalVars.h"
#include "devFunctionProtos.h"
#include "cudaRandFuncs.cu" /* nvcc doesn't compile without the source !*/
#include "math.h"
/* ff input */
__global__ void AuxRffTotal(curandState *devNormRandState, curandState *devStates) {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x ;
  int i;
  if(mNeuron < N_Neurons) {
    randnXiA[mNeuron] =  normRndKernel(devNormRandState);
    randuDelta[mNeuron] = PI * randkernel(devStates);
    for(i = 0; i < 4; ++i) {
      randwZiA[mNeuron * 4 + i] = 1.4142135 * sqrt(-1.0 * log(randkernel(devStates)));
    }
    for(i = 0; i < 3; ++i) {
      randuPhi[mNeuron * 3 + i] = 2.0 * PI * randkernel(devStates);
    }
  }
  }


__device__ void RffTotal(double t) {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  float varContrast;
  varContrast = CONTRAST;
  if(mNeuron < N_Neurons) {
    /*  
    if(t < 3000) { // SWITCH ON STIMULUS AT 5000ms 
      varContrast = 0.0;
    }
    else {
      varContrast = 100.0;
    }
    */
    if(mNeuron < NE) {
      /**  !!!!! COS IN RADIANS ?????? */
      rTotal[mNeuron] = CFFE * K * (R0 +  R1 * log10(1.000 + varContrast))
	+ sqrt(CFFE * K) * R0 * randnXiA[mNeuron]
	+ sqrt(CFFE * K) * R1 * log10(1.0 + varContrast) * (randnXiA[mNeuron] 
		      + ETA_E * randwZiA[mNeuron * 4] * cos(2.0 * (theta - randuDelta[mNeuron])) 
		      + MU_E * randwZiA[mNeuron *4 + 1] * cos(INP_FREQ * t - randuPhi[mNeuron * 3])
		      + ETA_E * MU_E * 0.5 * (randwZiA[mNeuron * 4 + 2] * cos(2.0 * theta + INP_FREQ * t - randuPhi[mNeuron * 3 + 1]) + randwZiA[mNeuron * 4 + 3] * cos(2.0 * theta - INP_FREQ * t + randuPhi[mNeuron * 3 + 2])));
    }
    if(mNeuron >= NE) {
      /*      rTotalPrev[mNeuron] = rTotal[mNeuron]; */

      rTotal[mNeuron] = CFFI * K * (R0 + R1 * log10(1.000 + varContrast)) + sqrt(CFFI * K) * R0 * randnXiA[mNeuron] + sqrt(CFFI * K) * R1 * log10(1.0 + varContrast) * (randnXiA[mNeuron] + ETA_I * randwZiA[mNeuron * 4] * cos(2.0 * (theta - randuDelta[mNeuron])) + MU_I * randwZiA[mNeuron * 4 + 1] * cos(INP_FREQ * t - randuPhi[mNeuron * 3]) + ETA_I * MU_I * 0.5 * (randwZiA[mNeuron * 4 + 2] * cos(2.0 * theta + INP_FREQ * t - randuPhi[mNeuron * 3 + 1]) + randwZiA[mNeuron * 4 + 3] * cos(2.0 * theta - INP_FREQ * t + randuPhi[mNeuron * 3 + 2])));
    }
    /*
      rTotal[mNeuron] = (R0 + R1 * log10(1 + CONTRAST)) * (CFF * K + sqrt(CFF *K) * randnXiA[mNeuron]);*/
  }
}
 

__device__ void Gff(double t) {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  double tmp = 0.0;
  if(mNeuron < N_Neurons) {
    if(t > DT) {
      tmp = gffItgrl[mNeuron];
      if(mNeuron < NE) {
	//        tmp = tmp * (1 - DT / TAU_SYNAP_E) + (SQRT_DT * INV_TAU_SYNAP_E) * normRndKernel(iffNormRandState);
        tmp = tmp * (1 - DT / TAU_FF) + (SQRT_DT * INV_TAU_FF) * normRndKernel(iffNormRandState);
        gFF[mNeuron] =   GFF_E * (rTotal[mNeuron] + sqrt(rTotal[mNeuron]) * tmp);
      }
      if(mNeuron >= NE) {
	//     tmp = tmp * (1 - DT / TAU_SYNAP_I) + (SQRT_DT * INV_TAU_SYNAP_I) * normRndKernel(iffNormRandState);
	tmp = tmp * (1 - DT / TAU_FF) + (SQRT_DT * INV_TAU_FF) * normRndKernel(iffNormRandState);
        gFF[mNeuron] =  GFF_I * (rTotal[mNeuron] + sqrt(rTotal[mNeuron]) * tmp);
      }
      gffItgrl[mNeuron] = tmp;
    }
    else {
      gffItgrl[mNeuron] = 0.0;
      gFF[mNeuron] = 0.0;
    }
  }
}

__device__ double IFF(double vm) {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  double iff = 0.0;
  if(mNeuron < N_Neurons) {
    iff = -1 * gFF[mNeuron] * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
    if(mNeuron == SAVE_CURRENT_FOR_NEURON) {
     dev_iff[curConter - 1] = iff;
    }
  }
  return iff;
}
#endif
