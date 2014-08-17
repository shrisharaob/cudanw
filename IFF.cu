#ifndef _IFFCURRENT_
#define _IFFCURRENT_
#include "globalVars.h"
#include "devFunctionProtos.h"
#include "cudaRandFuncs.cu" /* nvcc doesn't compile without the source !*/
#include "math.h"
/* ff input */
__global__ void AuxRffTotal(curandState *devNormRandState, curandState *devStates) {
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x ;
  int i;
  if(mNeuron < N_Neurons) {
    randnXiA[mNeuron] =  normRndKernel(devNormRandState);
    randuDelta[mNeuron] = PI * randkernel(devStates);
    for(i = 0; i < 4; ++i) {
      randwZiA[mNeuron * 4 + i] = 1.4142135 * sqrt(-1 * log(randkernel(devStates)));
    }
    for(i = 0; i < 3; ++i) {
      randuPhi[mNeuron * 3 + i] = 2 * PI * randkernel(devStates);
    }
  }
  }


__device__ void RffTotal(double t) {
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  if(mNeuron < N_Neurons) {
    if(mNeuron < NE) {
      /*    rTotalPrev[mNeuron] = rTotal[mNeuron]; */
      rTotal[mNeuron] = CFF * K * (R0 + R1 * log10(1 + CONTRAST)) 
	+ sqrt(CFF * K) * R0 * randnXiA[mNeuron]
	+ sqrt(CFF * K) * R1 * log10(1 + CONTRAST) * (randnXiA[mNeuron] 
		      + ETA_E * randwZiA[mNeuron * 4] * cos(2 * (theta - randuDelta[mNeuron])) 
		      + MU_E * randwZiA[mNeuron *4 + 1] * cos(INP_FREQ * t - randuPhi[mNeuron * 3])
		      + ETA_E * MU_E * 0.5 * (randwZiA[mNeuron * 4 + 2] * cos(2 * theta + INP_FREQ * t - randuPhi[mNeuron * 3 + 1]) + randwZiA[mNeuron * 4 + 3] * cos(2 * theta - INP_FREQ * t + randuPhi[mNeuron * 3 + 2])));
    }
    if(mNeuron >= NE) {
      /*      rTotalPrev[mNeuron] = rTotal[mNeuron]; */
      rTotal[mNeuron] = CFF * K * (R0 + R1 * log10(1 + CONTRAST)) 
	+ sqrt(CFF * K) * R0 * randnXiA[mNeuron]
	+ sqrt(CFF * K) * R1 * log10(1 + CONTRAST) * (randnXiA[mNeuron] 
		     		      + ETA_I * randwZiA[mNeuron * 4] * cos(2 * (theta - randuDelta[mNeuron])) 
				      + MU_I * randwZiA[mNeuron * 4 + 1] * cos(INP_FREQ * t - randuPhi[mNeuron * 3])
				      + ETA_I * MU_I * 0.5 * (randwZiA[mNeuron * 4 + 2] 
					      * cos(2 * theta + INP_FREQ * t - randuPhi[mNeuron * 3 + 1])
				      + randwZiA[mNeuron * 4 + 3] 
				      * cos(2 * theta - INP_FREQ * t + randuPhi[mNeuron * 3 + 2])));

    }
    if(mNeuron == 10) {
      dev_iff[curConter - 1] = rTotal[mNeuron];
    }
  }
}
 

__device__ void Gff(double t) {
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  float Itgrl;
  if(mNeuron < N_Neurons) {
    if(t > DT) {
      if(mNeuron < NE) {
	Itgrl = rTotal[mNeuron] + sqrt(rTotal[mNeuron]) * normRndKernel(iffNormRandState);
	gFF[mNeuron] += DT * (-1 * GFF_E * sqrt(1/K) * INV_TAU_SYNAP 
			      * ( INV_TAU_SYNAP * gFF[mNeuron] - Itgrl));
      }
      if(mNeuron >= NE) {
	Itgrl = rTotal[mNeuron] + sqrt(rTotal[mNeuron]) * normRndKernel(iffNormRandState); /*/SQRT_DT;*/
	gFF[mNeuron] += DT * (-1 * GFF_I * sqrt(1/K) * INV_TAU_SYNAP 
			      * ( INV_TAU_SYNAP * gFF[mNeuron] - Itgrl));
      }
    }
    /*    else {
      for(mNeuron = 1; mNeuron <= NE; ++mNeuron) {
    	Itgrl[mNeuron] = rTotal[mNeuron] + sqrt(rTotal[mNeuron]) * gaxosdev(&idem); // / SQRT_DT;
      }  
      for(mNeuron = NE + 1; mNeuron <= N_Neurons; ++mNeuron) {
    	Itgrl[mNeuron] = rTotal[mNeuron] + sqrt(rTotal[mNeuron]) * normRndKernel(&idem); // / SQRT_DT;
      }
      }
    */
  }
}

__device__ double IFF(double vm) {
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  double iff = 0;
  if(mNeuron < N_Neurons) {
    iff = -1 * gFF[mNeuron] * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
  }
  /*if(mNeuron == 0) {
    dev_iff[curConter - 1] = rTotal[mNeuron];
    }*/
  return iff;
}
#endif
