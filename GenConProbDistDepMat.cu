#ifndef _CONNECTION_PROB_
#define _CONNECTION_PROB_
#include <stdio.h>
#include <cuda.h>


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

__device_ double YCordinate(unsigned long int neuronIdx) {
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
      if((float)conProb(xa, ya, XCordinate(i), YCordinate(i)) >= randkernel(state, kNeuron)) { 
        dev_conVec[mNeuron + i * N_NEURONS] = 1;
      }
    }
    mNeuron += stride;
  }
}

__global__ void KernelConProbPreFactor(float *dev_conVec) {
  // compute pre-factor zB[clm] = K / sum(conProd(:, clm))
  unsigned long mNeuron = (unsigned long)(threadIdx.x + blockIdx.x * blockDim.x);
  unsigned long int i;
   int stride = gridDim.x * blockDim.x;
 

 for(clmId = 1; clmId <= N_Neurons; ++clmId) {
    for(rowId = 1; rowId <= NE; ++rowId) {
      zE[clmId] += conProb[rowId][clmId];
    }
    zE[clmId] = (double)K / zE[clmId];
    for(rowId = NE + 1; rowId <= N_Neurons; ++rowId) {
      zI[clmId] += conProb[rowId][clmId];
    }
    //    printf("%f ", zI[clmId]);
    zI[clmId] =(double)K /  zI[clmId];
    //    printf("%f \n", zI[clmId]);
  }
}

void genConMat() {
  double **conProb, *zE, *zI, xDiff, yDiff, z1, denom, ranttt, tempZI;
  int clmId, rowId, IF_CONNECT;
  long idum = -1 * rand();
  FILE *conProbFP, *conMatFP;
  strcpy(filebase, FILEBASE);
  conProbFP = fopen(strcat(filebase,"conProbMat"), "w");
  strcpy(filebase, FILEBASE);
  conMatFP = fopen(strcat(filebase,"conMatFp"), "w");
  strcpy(filebase, FILEBASE);
  conProb = matrix(1, N_Neurons, 1, N_Neurons);
  strcpy(filebase, FILEBASE);
  zI = vector(1, N_Neurons);
  zE = vector(1, N_Neurons);
  z1 =   (1 / sqrt(2 * PI * CON_SIGMA));
  denom = (2 * CON_SIGMA * CON_SIGMA);
  // connection probablity for E cells
  for(rowId =1; rowId <= NE; ++rowId) {
    for(clmId = 1; clmId <= NE; ++clmId) {
      xDiff = XCordinate(rowId, NE) - XCordinate(clmId, NE);
      yDiff = YCordinate(rowId, NE) - YCordinate(clmId, NE);
      conProb[rowId][clmId] =  z1 * z1 * exp(-1 * pow(fmod(xDiff, L), 2) / (denom)) * z1 * z1 * exp(-1 * pow(fmod(yDiff, L), 2) / (denom));
      fprintf(conProbFP ,"%f ", conProb[rowId][clmId]);
    }
    for (clmId = NE + 1; clmId <= N_Neurons; ++clmId) {
      xDiff = XCordinate(rowId, NE) - XCordinate(clmId - NE, NI);
      yDiff = YCordinate(rowId, NE) - YCordinate(clmId - NE, NI);
      conProb[rowId][clmId] =  z1 * z1 * exp(-1 * pow(fmod(xDiff, L), 2) / (denom))
        * z1 * z1 * exp(-1 * pow(fmod(yDiff, L), 2) / (denom));
      fprintf(conProbFP, "%f ", conProb[rowId][clmId]);
    }
    fprintf(conProbFP, "\n");
  }
  // connection probablity for I cells
  for(rowId = 1 + NE; rowId <= N_Neurons; ++rowId) {
    for(clmId = 1; clmId <= NE; ++clmId) {
      xDiff = XCordinate(rowId - NE, NI) - XCordinate(clmId, NE);
      yDiff = YCordinate(rowId - NE, NI) - YCordinate(clmId, NE);
      conProb[rowId][clmId] = z1 * z1 * exp(-1 * pow(fmod(xDiff, L), 2) / (denom))
        * z1 * z1 * exp(-1 * pow(fmod(yDiff, L), 2) / (denom));
      fprintf(conProbFP ,"%f ", conProb[rowId][clmId]);
    }
    for (clmId = NE + 1; clmId <= N_Neurons; ++clmId) {
      xDiff = XCordinate(rowId - NE, NI) - XCordinate(clmId - NE, NI);
      yDiff = YCordinate(rowId - NE, NI) - YCordinate(clmId - NE, NI);
      conProb[rowId][clmId] = z1 * z1 * exp(-1 * pow(fmod(xDiff, L), 2) / (denom))
        * z1 * z1 * exp(-1 * pow(fmod(yDiff, L), 2) / (denom));
      fprintf(conProbFP, "%f ", conProb[rowId][clmId]);
    }
    fprintf(conProbFP, "\n");
  }
  fclose(conProbFP);
  // compute pre-factor zB[clm] = K / sum(conProd(:, clm))
  for(clmId = 1; clmId <= N_Neurons; ++clmId) {
    for(rowId = 1; rowId <= NE; ++rowId) {
      zE[clmId] += conProb[rowId][clmId];
    }
    //printf("b - %f ", zE[clmId]);
    zE[clmId] = (double)K / zE[clmId];
    //printf("a - %f \n", zE[clmId]);
    for(rowId = NE + 1; rowId <= N_Neurons; ++rowId) {
      zI[clmId] += conProb[rowId][clmId];
    }
    //printf("%f ", zI[clmId]);
    zI[clmId] =(double)K /  zI[clmId];
    //printf("%f \n", zI[clmId]);
  }
  // randomly connect neurons with probability given by conProb
  for(rowId = 1; rowId <= NE; ++rowId) {
    for(clmId = 1; clmId <= N_Neurons; ++clmId) {
      //printf("%f %ld \n", ranttt, idum);
      if(ran2(&idum) <=  zE[clmId] * conProb[rowId][clmId]) {
        conMat[rowId][clmId] = 1;
      }
      fprintf(conMatFP ,"%f ", conMat[rowId][clmId]);
    }
    fprintf(conMatFP, "\n");
  }
  srand(time(NULL));  
  for(rowId = 1 + NE; rowId <= N_Neurons; ++rowId) {
    for(clmId = 1; clmId <= N_Neurons; ++clmId) {
      //  printf("%ld \n", idum);
      if(ran2(&idum) <=  zI[clmId] * conProb[rowId][clmId]) {
        conMat[rowId][clmId] = 1;
      }
      fprintf(conMatFP,"%f ", conMat[rowId][clmId]);
    }
    fprintf(conMatFP,"\n");
  }
  fclose(conMatFP);
}
#endif