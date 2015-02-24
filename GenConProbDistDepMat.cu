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

__device__ double Gaussian2D(double x, double y) {
  double z1 (1 / sqrt(2 * PI * CON_SIGMA)); // make global var
  double denom = (2 * CON_SIGMA * CON_SIGMA); // global var
  return  z1 * z1 * exp(-1 * pow(x, 2) / (denom)) * z1 * z1 * exp(-1 * pow(y, 2) / (denom));
}


__device__ double Gaussian2D(double x, double y, double varianceOfGaussian) {
  double z1 (1 / sqrt(2 * PI * varianceOfGaussian)); // make global var
  double denom = (2 * varianceOfGaussian * varianceOfGaussian); // global var
  return  z1 * z1 * exp(-1 * pow(x, 2) / (denom)) * z1 * z1 * exp(-1 * pow(y, 2) / (denom));
}

__device__ double ShortestDistOnCirc(double point0, double point1, double perimeter) {
  double dist = 0.0;
  dist = abs(point0 - point1);
  dist = fmod(dist, perimeter);
  if(dist > 0.5){
    dist = 1.0 - dist;
  }
  return dist;
}


__device__ double ConProb_new(double xa, double ya, double xb, double yb, double patchSize, double varianceOfGaussian) {
  double distX = 0.0; //ShortestDistOnCirc(xa, xb, patchSize);
  double distY = 0.0; //ShortestDistOnCirc(ya, yb, patchSize);
  double out = 0.0;
  int IF_PERIODIC = 1;
  if(IF_PERIODIC) {
    distX = ShortestDistOnCirc(xa, xb, patchSize);
    distY = ShortestDistOnCirc(ya, yb, patchSize);
  }
  else {
    distX = abs(xa - xb);
    distY = abs(ya - yb);
  }
  return Gaussian2D(distX, distY, varianceOfGaussian);
}


__device__ double conProb(double xa, double ya, double xb, double yb) {
  /* returns connection probablity given cordinates (xa, ya) and (xb, yb) */
  /*  double z1 (1 / sqrt(2 * PI * CON_SIGMA)); // make global var
  double denom = (2 * CON_SIGMA * CON_SIGMA); // global var*/
  double x0, x1, y0, y1, result; 
  x0 = fmod(abs(xa-xb), L);
  x1 = fmod(abs(xa-xb), -1 * L);
  y0 = fmod(abs(ya-yb), L);
  y1 = fmod(abs(ya-yb), -1 * L);
  result = Gaussian2D(x0, y0) + Gaussian2D(x1, y1);
  if(x0 == 0 && x1 == 0) {
    result *= sqrt(0.5);
  }
  if(y0 == 0 && y1 == 0) {
    result *= sqrt(0.5);
  }
  /*  return z1 * z1 * exp(-1 * pow(fmod(xa - xb, L), 2) / (denom)) * z1 * z1 * exp(-1 * pow(fmod(ya - yb, L), 2) / (denom));*/
  return result;
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
      //dev_conVec[mNeuron + i * N_NEURONS] = (float)conProb(xa, ya, XCordinate(i), YCordinate(i)); 
      dev_conVec[mNeuron + i * N_NEURONS] = (float)ConProb_new(xa, ya, XCordinate(i), YCordinate(i), L, CON_SIGMA); 
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