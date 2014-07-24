#ifndef _DERIVS_
#define _DERIVS_

#include "globalVars.h"
#include "devFunctionProtos.h"
#include <cuda.h>

__device__ float alpha_n(float vm);
__device__ float alpha_m(float vm);
__device__ float alpha_h(float vm);
__device__ float beta_n(float vm);
__device__ float beta_m(float vm);
__device__ float beta_h(float vm);

__device__ float alpha_n(float vm) {
  float out;
  if(vm != -34) { 
    out = 0.1 * (vm + 34) / (1 - exp(-0.1 * (vm + 34)));
  }
  else {
    out = 0.1;
  }
  return out;
}

__device__ float beta_n(float vm) {
  float out;
  out = 1.25 * exp(- (vm + 44) / 80);
  return out;
}

__device__ float alpha_m(float vm) {
  float out;
  if(vm != -30) { 
    out = 0.1 * (vm + 30) / (1 - exp(-0.1 * (vm + 30)));
  }
  else {
    out = 1;
  }
  return out;
}

__device__ float beta_m(float vm) {
  float out;
  out = 4 * exp(-(vm + 55) / 18);
  return out;
}

__device__ float alpha_h(float vm) {
  float out;
  out = 0.7 * exp(- (vm + 44) / 20);
  return out;
}

__device__ float beta_h(float vm) {
  float out;
  out = 10 / (exp(-0.1 * (vm + 14)) + 1);
  return out;
  }

__device__ float m_inf(float vm) {
  float out, temp;
  temp = alpha_m(vm);
  out = temp / (temp + beta_m(vm));
  return out;
}

//z is the gating varible of the adaptation current
__device__ float z_inf(float(vm)) {
  float out;
  out = 1 / (1 + exp(-0.7 *(vm + 30)));
  return out;
}


//=========================================================================================\\

//extern float dt, *iSynap;
// stateVar = [vm, n, z, h]
// z - gating variable of the adaptation current
__device__ void derivs(float t, float stateVar[], float dydx[], float isynap, float ibg, float iff) {
  float cur = 0;
  int kNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  float bgPrefactor = 0.0, iffPrefactor = 0.0;
  if(kNeuron < N_NEURONS) {
    cur = 0.1 * sqrt(K);
    cur = 2.8;
    /*if((kNeuron == 13520 & t >= 30 & t <= 35) | (kNeuron == 2 & t >= 50 & t <= 55)) {cur = 10;}*/
    /*    if(kNeuron >= 13520) {
      cur = 3.0;
      }*/

    if (kNeuron < NE) {
      dydx[0] =  1/Cm * (cur 
                                 - G_Na * pow(m_inf(stateVar[0]), 3) * stateVar[3] * (stateVar[0] - E_Na) 
                                 - G_K * pow(stateVar[1], 4) * (stateVar[0] - E_K) 
                                 - G_L_E * (stateVar[0] - E_L)
			 - G_adapt * stateVar[2] * (stateVar[0] - E_K) + isynap + bgPrefactor * ibg + iffPrefactor * iff);
      }
    else {
      dydx[0] =  1/Cm * (cur  
                                   - G_Na * pow(m_inf(stateVar[0]), 3) * stateVar[3] * (stateVar[0] - E_Na) 
                                   - G_K * pow(stateVar[1], 4) * (stateVar[0] - E_K) 
                                   - G_L_I * (stateVar[0] - E_L)
		       - G_adapt * stateVar[2] * (stateVar[0] - E_K) + isynap + bgPrefactor * ibg + iffPrefactor * iff);
      }
     
  dydx[1] = alpha_n(stateVar[0]) * (1 - stateVar[1]) - beta_n(stateVar[0]) * stateVar[1];
  
  dydx[2] = 1 / Tau_adapt * (z_inf(stateVar[0]) - stateVar[2]);
    
  dydx[3] = alpha_h(stateVar[0]) * (1 - stateVar[3]) - beta_h(stateVar[0]) * stateVar[3];
 
  }
}

#endif
