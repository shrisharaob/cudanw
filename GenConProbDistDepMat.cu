#ifndef _CONNECTION_PROB_
#define _CONNECTION_PROB_
#include <stdio.h>
#include <cuda.h>
#include "devHostConstants.h"

/* GENERATE CONNECTION MATRIX */
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


__device__ double conProb(double xa, double ya, double xb, double yb) {
  /* returns connection probablity given cordinates (xa, ya) and (xb, yb) */
  double z1 (1 / sqrt(2 * PI * CON_SIGMA)); // make global var
  double denom = (2 * CON_SIGMA * CON_SIGMA); // global var
  return z1 * z1 * exp(-1 * pow(fmod(xa - xb, L), 2) / (denom)) * z1 * z1 * exp(-1 * pow(fmod(ya - yb, L), 2) / (denom));
}

__global__ void KernelGenConProbMat(float *dev_conVec) {
  unsigned long mNeuron = (unsigned long)(threadIdx.x + blockIdx.x * blockDim.x);
  unsigned long int i;
  double xa, ya;
  int stride = gridDim.x * blockDim.x;
  while(mNeuron < N_NEURONS) {
    xa = XCordinate(mNeuron);
    ya = YCordinate(mNeuron);
    for(i = 0; i < N_NEURONS; ++i) {
      dev_conVec[mNeuron + i * N_NEURONS] = (float)conProb(xa, ya, XCordinate(i), YCordinate(i)); 
    }
    mNeuron += stride;
  }
}

__global__ void KernelConProbPreFactor(float *dev_conVec) {
  /*  COMPUTE PRE-FACTOR AND MULTIPLY zB[clm] = K / sum(conProd(:, clm)) */
  unsigned long mNeuron = (unsigned long)(threadIdx.x + blockIdx.x * blockDim.x); // each column is a thread
  unsigned long int i;
  double preFactorE2All, preFactorI2All;
  int stride = gridDim.x * blockDim.x;
  while(mNeuron < N_NEURONS) {
    preFactorI2All = 0.0;
    preFactorE2All = 0.0;
    for(i = 0; i < N_NEURONS; ++i) { // sum over rows
      if(i < NE) {
        preFactorE2All += (double)dev_conVec[i + mNeuron * N_NEURONS];
      }
      else {
        preFactorI2All += (double)dev_conVec[i + mNeuron * N_NEURONS];
      }
    }     
    preFactorI2All = (double)K / preFactorI2All;
    preFactorE2All = (double)K / preFactorE2All;
    /* now multiply the prefactor */
    for(i = 0; i < N_NEURONS; ++i) { 
      if(i < NE) {
        dev_conVec[i + mNeuron * N_NEURONS] *= (float)preFactorE2All;
      }
      else {
        dev_conVec[i + mNeuron * N_NEURONS] *= (float)preFactorI2All;
      }
    }     
    mNeuron += stride;
  }
}


#endif