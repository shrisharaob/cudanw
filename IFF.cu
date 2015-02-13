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
  double out = 0.0;
  //int angleId;
  //  double minAngle, maxAngle;
  if(mNeuron < N_Neurons) {
    randnXiA[mNeuron] =  normRndKernel(devNormRandState);
    if(IF_ORI_MAP) {
      out = OrientationAngleForGivenNeuron(mNeuron);
      if(out == 720.0){randuDelta[mNeuron] = PI * randkernel(devStates);} // this is to ensure that the POs have a uniform distr
      else {randuDelta[mNeuron] = out;}
      // angleId = OrientationForGivenNeuron(mNeuron);
      // minAngle = angleId * (PI / 8.0);
      // maxAngle = (angleId+1) * (PI / 8.0);
      // randuDelta[mNeuron] = (maxAngle - minAngle) * randkernel(devStates) + minAngle;
    }
    else {
      randuDelta[mNeuron] = PI * randkernel(devStates);
    }
    for(i = 0; i < 4; ++i) {
      randwZiA[mNeuron * 4 + i] = 1.4142135 * sqrt(-1 * log(randkernel(devStates)));
    }
    for(i = 0; i < 3; ++i) {
      randuPhi[mNeuron * 3 + i] = 2 * PI * randkernel(devStates);
    }
  }
}

__device__ double XCordinate(unsigned long int neuronIdx) {
  // nA - number of E or I cells
  double nA = (double)NE;
  if(neuronIdx > NE) { // since neuronIds for inhibitopry start from NE jusqu'a N_NEURONS
    neuronIdx -= NE;
    nA = (double)NI;
  }
  return fmod((double)neuronIdx, sqrt(nA)) * (L / (sqrt(nA) - 1));
}

__device__ double YCordinate(unsigned long  neuronIdx) {
  double nA = (double)NE;
  if(neuronIdx > NE) {
    neuronIdx -= NE;
    nA = (double)NI;
  }
  return floor((double)neuronIdx / sqrt(nA)) * (L / (sqrt(nA) - 1));   
}

__device__ double MyDivide(double x, double y) {
  // RETURNS x/y CONSIDERING CASES WHEN y = 0
  double out = 0.0;
  if(y == 0.0)  {out = 1E10;}
  else {out = x/y; }
  return out;
}

__device__ double OrientationAngleForGivenNeuron(unsigned int neuronId){
  // SIMPLE PIN WHEEL PO ASSIGNMENT BASED ON LOCATION ON THE PATCH 
  unsigned long mNeuron;
  double xCordinate = 0.0, yCordinate = 0.0;
  double out  = 0;
  int IF_CIRCLE = 0;
  mNeuron = (unsigned long) neuronId;
  xCordinate = XCordinate(mNeuron);
  yCordinate = YCordinate(mNeuron);
  //pinwheel center coincides with the center of patch, so shift origin to center of patch
  xCordinate = xCordinate - (L * 0.5);
  yCordinate = yCordinate - (L * 0.5);
  if(IF_CIRCLE) {
    if((xCordinate*xCordinate) + (yCordinate * yCordinate) <= (L * 0.5) * (L * 0.5)) {
    // if neuron lies inside the circle of radius 
    out = fmod(atan(MyDivide(yCordinate, xCordinate)) + PI, PI); 
    }
    else {
      out = 720.0;
    }
  }
  else  {
    out = fmod(atan(MyDivide(yCordinate, xCordinate)) + PI, PI); 
  }
  return out;
}

__device__ int OrientationForGivenNeuron(unsigned int neuronId){
  // SIMPLE PIN WHEEL PO ASSIGNMENT BASED ON LOCATION ON THE PATCH 
  unsigned long mNeuron;
  int quadrant = 0, angleId = 0;
  double xCordinate = 0.0, yCordinate = 0.0;
  mNeuron = (unsigned long) neuronId;
  xCordinate = XCordinate(mNeuron);
  yCordinate = YCordinate(mNeuron);
  //pinwheel center coincides with the center of patch, so shift origin to center of patch
  xCordinate = xCordinate - (L * 0.5);
  yCordinate = yCordinate - (L * 0.5);
  //get quadrant
  if(yCordinate >= 0) {
    if(xCordinate < 0) {
      quadrant = 1;
    }
  }
  else {
    if(xCordinate < 0) {
      quadrant = 2;
    }
    else {
      quadrant = 3;
    }
  }
  //get angleId which is the id of the sector
  switch(quadrant) {
  case 0:
    if(yCordinate <= xCordinate) {angleId = 0;}
    else {angleId = 1;}
    break;
  case 1:
    if(yCordinate <= (-1 * xCordinate)) {angleId = 3;}
    else {angleId = 2;}
    break;
  case 2:
    if(yCordinate <= xCordinate) {angleId = 5;}
    else {angleId = 4;}
    break;
  case 3:
    if(yCordinate <= (-1 * xCordinate)) {angleId = 6;}
    else {angleId = 0;} //0 in stead of 7
  }
  return angleId;
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

      rTotal[mNeuron] = CFF * K * (R0 +  R1 * log10(1.000 + varContrast))
	+ sqrt(CFF * K) * R0 * randnXiA[mNeuron]
	+ sqrt(CFF * K) * R1 * log10(1.0 + varContrast) * (randnXiA[mNeuron] 
		      + ETA_E * randwZiA[mNeuron * 4] * cos(2.0 * (theta - randuDelta[mNeuron])) 
		      + MU_E * randwZiA[mNeuron *4 + 1] * cos(INP_FREQ * t - randuPhi[mNeuron * 3])
		      + ETA_E * MU_E * 0.5 * (randwZiA[mNeuron * 4 + 2] * cos(2.0 * theta + INP_FREQ * t - randuPhi[mNeuron * 3 + 1]) + randwZiA[mNeuron * 4 + 3] * cos(2.0 * theta - INP_FREQ * t + randuPhi[mNeuron * 3 + 2])));
    }
    if(mNeuron >= NE) {
      /*      rTotalPrev[mNeuron] = rTotal[mNeuron]; */

      rTotal[mNeuron] = CFF * K * (R0 + R1 * log10(1.000 + varContrast)) + sqrt(CFF * K) * R0 * randnXiA[mNeuron] + sqrt(CFF * K) * R1 * log10(1.0 + varContrast) * (randnXiA[mNeuron] + ETA_I * randwZiA[mNeuron * 4] * cos(2.0 * (theta - randuDelta[mNeuron])) + MU_I * randwZiA[mNeuron * 4 + 1] * cos(INP_FREQ * t - randuPhi[mNeuron * 3]) + ETA_I * MU_I * 0.5 * (randwZiA[mNeuron * 4 + 2] * cos(2.0 * theta + INP_FREQ * t - randuPhi[mNeuron * 3 + 1]) + randwZiA[mNeuron * 4 + 3] * cos(2.0 * theta - INP_FREQ * t + randuPhi[mNeuron * 3 + 2])));
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
      tmp = tmp * (1 - DT / TAU_SYNAP) + (SQRT_DT * INV_TAU_SYNAP) * normRndKernel(iffNormRandState);
      if(mNeuron < NE) {
        gFF[mNeuron] =   GFF_E * (rTotal[mNeuron] + sqrt(rTotal[mNeuron]) * tmp);
      }
      if(mNeuron >= NE) {
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
    if(mNeuron == SAVE_CURRENT_FOR_NEURON) { dev_iff[curConter - 1] = iff;}
  }
  return iff;
}
#endif
