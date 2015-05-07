#ifndef _RKDUMBPRETTY_
#define _RKDUMBPRETTY_
#include <cuda.h>
#include "globalVars.h"
#include "devFunctionProtos.h"
#include "derivs.cu"
#include "rk4.cu"

__global__ void rkdumbPretty(kernelParams_t params, devPtr_t devPtrs) { 
  double x1, *dev_spkTimes, *y,  *synapticCurrent, *dev_time;

  int nstep, *totNSpks, *dev_spkNeuronIds;
  curandState *dev_state;
  int k;
  double x, isynapNew = 0, ibg = 0, iff = 0;
  double v[N_STATEVARS], vout[N_STATEVARS], dv[N_STATEVARS], vmOld;
  int localTotNspks = 0, localLastNSteps;
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  int storedvmIdx;
  x1 = params.tStart;
  nstep = params.nSteps;
  totNSpks = devPtrs.dev_nSpks;
  y = devPtrs.dev_vm;
  dev_time = devPtrs.dev_time;
  synapticCurrent = devPtrs.synapticCurrent;
  dev_state = devPtrs.devStates;
  dev_spkTimes = devPtrs.dev_spkTimes;
  dev_spkNeuronIds = devPtrs.dev_spkNeuronIds;
  /*  dev_nPostNeurons = devPtrs.dev_nPostNeurons;
  dev_sparseConVec = devPtrs.dev_sparseConVec;
  dev_sparseIdx = devPtrs.dev_sparseIdx;*/
  k = devPtrs.k;
  if(mNeuron < N_NEURONS) {
    if(k == 0) {
      dev_v[mNeuron] = (-1 * 70) +  (40 * randkernel(dev_state)); /* Vm(0) ~ U(-70, -30)*/
      dev_n[mNeuron] = randkernel(dev_state); //0.3176;
      dev_z[mNeuron] = randkernel(dev_state); //0.1;
      dev_h[mNeuron] = randkernel(dev_state); //0.5961;
      dev_isynap[mNeuron] = 0;
      dev_gE[mNeuron] = 0.0;
      dev_gI[mNeuron] = 0.0;
      dev_sparseConVec = devPtrs.dev_sparseConVec;
      dev_sparseConVecFF = devPtrs.dev_sparseConVecFF;
      if(mNeuron < NE) {
        gaussNoiseE[mNeuron] = 0.0;
      }
      else {
        gaussNoiseI[mNeuron - NE] = 0.0;
      }
      gFF[mNeuron] = 0.0; 
      rTotal[mNeuron] = 0.0;
      gffItgrl[mNeuron] = 0.0; // holds the old value of gff 
      dev_GFFmean[mNeuron] = 0.0;
      devGFFCounter = 0;
    }
    localLastNSteps = nstep - STORE_LAST_N_STEPS;
    /* TIMELOOP */
    x = x1 + (double)k * DT;
    dev_IF_SPK[mNeuron] = 0;
    vmOld = dev_v[mNeuron];
    v[0] = vmOld;
    v[1] = dev_n[mNeuron];
    v[2] = dev_z[mNeuron];
    v[3] = dev_h[mNeuron];
    isynapNew = dev_isynap[mNeuron];
    iff = dev_iffCurrent[mNeuron];
    ibg = bgCur(vmOld);
    /* runge kutta 4 */
    derivs(x, v, dv, isynapNew, ibg, iff);
    rk4(v, dv, N_STATEVARS, x, DT, vout, isynapNew, ibg, iff);
    x += DT; 
    /* UPDATE */
    dev_v[mNeuron] = vout[0];
    dev_n[mNeuron] = vout[1];
    dev_z[mNeuron] = vout[2];
    dev_h[mNeuron] = vout[3];
    if(k >= localLastNSteps & (mNeuron >= N_NEURONS_TO_STORE_START &  mNeuron < N_NEURONS_TO_STORE_END)) {
      //      storedvmIdx = (int) ((double)k/4.0);
      //      if(storedvmIdx * 4 == k) {
	y[(mNeuron - N_NEURONS_TO_STORE_START) + N_NEURONS_TO_STORE * (k - localLastNSteps)] = vout[0];
	//	y[(mNeuron - N_NEURONS_TO_STORE_START) + N_NEURONS_TO_STORE * (storedvmIdx - localLastNSteps)] = vout[0];
	//      }
      /*      synapticCurrent[mNeuron + N_NEURONS *  (k - localLastNSteps)] = isynapNew;*/
	
	if(mNeuron == 0) {
	  dev_time[k - localLastNSteps] = x;
	}
    }
    if(k > 20000) { // which is 1 sec for dt = 0.05
      dev_GFFmean[mNeuron] += dev_gE[mNeuron]; //gFF[mNeuron] +
      if(mNeuron == 0) {
        devGFFCounter += 1;
      }
    }     
    if(mNeuron == SAVE_CURRENT_FOR_NEURON) {
      int localCurConter = curConter;
      if(localCurConter < N_CURRENT_STEPS_TO_STORE) {
        glbCurE[localCurConter] = iff + ibg;  // E
        glbCurI[localCurConter] = isynapNew; // I 
        dev_iff[localCurConter] = iff + ibg + isynapNew; //dev_iffCurrent[mNeuron];
        curConter += 1;
      }
    }


    
    /*    if(mNeuron = 0 & mNeuron <= N_E_2BLOCK_NA_CURRENT ) {
      synapticCurrent[mNeuron] = isynap + ibg + iff;
      }*/

    if(k > 500 &  mNeuron >= NE & mNeuron <= NE + N_I_SAVE_CUR) {
      synapticCurrent[mNeuron - NE] = isynapNew + ibg + iff;
    }

    if(k > 2) {
      if(vout[0] > SPK_THRESH) { 
	if(vmOld <= SPK_THRESH) {
	  dev_IF_SPK[mNeuron] = 1;
	  localTotNspks = atomicAdd(totNSpks, 1); /* atomic add on global introduces memory latency*/
	  if(localTotNspks + 1 < MAX_SPKS) {
	    dev_spkNeuronIds[localTotNspks + 1] = mNeuron;
	    dev_spkTimes[localTotNspks + 1] = x;
	  }
	}
      }
    }
  }
}
#endif
