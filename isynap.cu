#ifndef _ISYNAP_
#define _ISYNAP_
#include  <cuda.h>
#include "globalVars.h"
#include "devHostConstants.h"
#include "devFunctionProtos.h"
#define MAX_SPKS_PER_T_STEP 1000
__device__ float isynap(float vm, int *dev_conVec) {
  //
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
 int i, spkNeuronId[MAX_SPKS_PER_T_STEP], localNSpks = 0;
  float totIsynap = 0, gE, gI, tempCurE = 0, tempCurI = 0;
  /* compute squares of entries in data array */
  // !!!!! neurons ids start from ZERO  !!!!!! 
  if(mNeuron < N_NEURONS) {
    gE = dev_gE[mNeuron];
    gI = dev_gI[mNeuron];
    gE *= EXP_SUM;
    gI *= EXP_SUM;
    for(i = 0; i < N_NEURONS; ++i) {
      if(dev_IF_SPK[i]) { // too many global reads 
        spkNeuronId[localNSpks] = i; 
        localNSpks += 1; // nspks in prev step
      }
    }
    if(localNSpks > 0){
      for(i = 0; i < localNSpks; ++i) { //  
        if(spkNeuronId[i] < NE) {
          gE += dev_conVec[spkNeuronId[i] + N_NEURONS * mNeuron];
        }
        else {
          gI += dev_conVec[spkNeuronId[i] + N_NEURONS * mNeuron]; //optimize !!!! gEI_I
        }
      }
    }
    dev_gE[mNeuron] = gE;
    dev_gI[mNeuron] = gI;
    if(mNeuron < NE) {
      tempCurE = -1 *  gE * (1/sqrt(K)) * INV_TAU_SYNAP * G_EE
                          * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      tempCurI = -1 * gI * (1/sqrt(K)) * INV_TAU_SYNAP * G_EI
                          * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));
    }
    if(mNeuron >= NE){
      tempCurE = -1 * gE * (1/sqrt(K)) * INV_TAU_SYNAP * G_IE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      tempCurI = -1 * gI * (1/sqrt(K)) * INV_TAU_SYNAP * G_II * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));
    }
    totIsynap = tempCurE + tempCurI; 
  }
  return totIsynap;
}


__device__ float SparseIsynap(double vm, int *dev_nPostNeurons, int *dev_sparseConVec, int *dev_sparseIdx, int IF_SPK) {
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  int kNeuron;
  float totIsynap = 0, gE, gI, tempCurE = 0, tempCurI = 0;
  if(mNeuron < N_NEURONS) {
    dev_gE[mNeuron] *= EXP_SUM;
    dev_gI[mNeuron] *= EXP_SUM;
     if(IF_SPK) {  
      for(kNeuron = 0; kNeuron <= dev_nPostNeurons[mNeuron]; ++kNeuron) { 
        if(mNeuron < NE) {       
          atomicAdd(&dev_gE[dev_sparseConVec[dev_sparseIdx[mNeuron] + kNeuron]], 1.0f); /*WORKS ONLY ON CC >= 2.0 */
       }
        else
          atomicAdd(&dev_gI[dev_sparseConVec[dev_sparseIdx[mNeuron] + kNeuron]], 1.0f);
      }
    } 
    gE = dev_gE[mNeuron];
    gI = dev_gI[mNeuron];
    
    /*    dev_gE[mNeuron] = gE;
          dev_gI[mNeuron] = gI;*/
    if(mNeuron < NE) {
      tempCurE = -1 * gE * (1/sqrt(K)) * INV_TAU_SYNAP * G_EE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));

      tempCurI = -1 * gI * (1/sqrt(K)) * INV_TAU_SYNAP * G_EI * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));
    }
    if(mNeuron > NE){
      tempCurE = -1 * gE * (1/sqrt(K)) * INV_TAU_SYNAP * G_IE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      tempCurI = -1 * gI * (1/sqrt(K)) * INV_TAU_SYNAP * G_II * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));
    }
    totIsynap = tempCurE + tempCurI; 
  }
  return totIsynap;
}

//*/
#endif
