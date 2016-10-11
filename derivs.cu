#ifndef _DERIVS_
#define _DERIVS_

#include "globalVars.h"
#include "devFunctionProtos.h"
#include <cuda.h>

__device__ double alpha_n(double vm);
__device__ double alpha_m(double vm);
__device__ double alpha_h(double vm);
__device__ double beta_n(double vm);
__device__ double beta_m(double vm);
__device__ double beta_h(double vm);

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
      (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, 
		      __double_as_longlong(val + 
					   __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double alpha_n(double vm) {
  double out;
  if(vm != -34) { 
    out = 0.1 * (vm + 34) / (1 - exp(-0.1 * (vm + 34)));
  }
  else {
    out = 0.1;
  }
  return out;
}

__device__ double beta_n(double vm) {
  double out;
  out = 1.25 * exp(- (vm + 44) / 80);
  return out;
}

__device__ double alpha_m(double vm) {
  double out;
  if(vm != -30) { 
    out = 0.1 * (vm + 30) / (1 - exp(-0.1 * (vm + 30)));
  }
  else {
    out = 1;
  }
  return out;
}

__device__ double beta_m(double vm) {
  double out;
  out = 4 * exp(-(vm + 55) / 18);
  return out;
}

__device__ double alpha_h(double vm) {
  double out;
  out = 0.7 * exp(- (vm + 44) / 20);
  return out;
}

__device__ double beta_h(double vm) {
  double out;
  out = 10 / (exp(-0.1 * (vm + 14)) + 1);
  return out;
  }

__device__ double m_inf(double vm) {
  double out, temp;
  temp = alpha_m(vm);
  out = temp / (temp + beta_m(vm));
  return out;
}

//z is the gating varible of the adaptation current
__device__ double z_inf(double(vm)) {
  double out;
  out = 1 / (1 + exp(-0.7 *(vm + 30)));
  return out;
}


/*
  extern double dt, *iSynap;
  stateVar = [vm, n, z, h]
  z - gating variable of the adaptation current
*/
__device__ void derivs(double t, double stateVar[], double dydx[], double isynap, double ibg, double iff) {
  double cur = 0.0;
  unsigned int kNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  double bgPrefactor = 1.0, iffPrefactor = 1.0;
  if(kNeuron < N_NEURONS) {
    /*    cur = 0.1 * sqrt(K);*/
    /*if((kNeuron == 0 & t >= 30 & t <= 35) | (kNeuron == 1 & t >= 280 & t <= 285)) {cur = 10;} 
      else {cur = 0.0;}*/
    /*    if(kNeuron >= 13520) {
      cur = 3.0;
      }*/
    
    if (kNeuron < NE) {
      if(kNeuron < N_E_2BLOCK_NA_CURRENT) { /* BLOCK Na2+ CHANNEL FOR THESE NEURONS */
        dydx[0] =  1/Cm * (cur 
                           - G_K * pow(stateVar[1], 4) * (stateVar[0] - E_K) 
                           - G_L_E * (stateVar[0] - E_L)
                           - G_adapt * stateVar[2] * (stateVar[0] - E_K) + isynap + bgPrefactor * ibg + iffPrefactor * iff);
      }
      else {
        dydx[0] =  1/Cm * (cur 
                           - G_Na * pow(m_inf(stateVar[0]), 3) * stateVar[3] * (stateVar[0] - E_Na) 
                           - G_K * pow(stateVar[1], 4) * (stateVar[0] - E_K) 
                           - G_L_E * (stateVar[0] - E_L)
                           - G_adapt * stateVar[2] * (stateVar[0] - E_K) + isynap + bgPrefactor * ibg + iffPrefactor * iff);
      }
    }
    else {
      if(kNeuron >= NE & kNeuron < NE + N_I_2BLOCK_NA_CURRENT) { /* BLOCK Na2+ CHANNEL FOR THESE NEURONS */
        dydx[0] =  1/Cm * (cur  
                           - G_K * pow(stateVar[1], 4) * (stateVar[0] - E_K) 
                           - G_L_I * (stateVar[0] - E_L)
                           - 0.0 * G_adapt * stateVar[2] * (stateVar[0] - E_K) + isynap + bgPrefactor * ibg + iffPrefactor * iff);
      }
      else {
        dydx[0] =  1/Cm * (cur  
                           - G_Na * pow(m_inf(stateVar[0]), 3) * stateVar[3] * (stateVar[0] - E_Na) 
                           - G_K * pow(stateVar[1], 4) * (stateVar[0] - E_K) 
                           - G_L_I * (stateVar[0] - E_L)
                           - 0.0 * G_adapt * stateVar[2] * (stateVar[0] - E_K) + isynap + bgPrefactor * ibg + iffPrefactor * iff);
      }
    }
     
    dydx[1] = alpha_n(stateVar[0]) * (1 - stateVar[1]) - beta_n(stateVar[0]) * stateVar[1];
  
    dydx[2] = 1 / Tau_adapt * (z_inf(stateVar[0]) - stateVar[2]);
    
    dydx[3] = alpha_h(stateVar[0]) * (1 - stateVar[3]) - beta_h(stateVar[0]) * stateVar[3];
  }
}

#endif
