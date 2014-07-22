#ifndef _RKDUMBPRETTY_
#define _RKDUMBPRETTY_
#include <cuda.h>
#include "globalVars.h"
#include "devFunctionProtos.h"
#include "derivs.cu"
#include "rk4.cu"

__global__ void rkdumbPretty(kernelParams_t params, devPtr_t devPtrs) { 
  float x1, *spkTimes, *y,  *dev_isynap;
  int nstep, *totNSpks, *spkNeuronId, *dev_nPostNeurons, *dev_sparseConVec, *dev_sparseIdx;
  curandState *dev_state;
  int i, k, IF_SPK;
  float x, h, xx, isynapNew = 0, ibg = 0, iff = 0;
  float v[N_STATEVARS], vout[N_STATEVARS], dv[N_STATEVARS], vmOld;
  int localTotNspks = 0, localLastNSteps;
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  x1 = params.tStart;
  nstep = params.nSteps;
  totNSpks = devPtrs.dev_nSpks;
  y = devPtrs.dev_vm;
  dev_isynap = devPtrs.dev_isynap;
  dev_state = devPtrs.devStates;
  spkTimes = devPtrs.dev_spkTimes;
  spkNeuronId = devPtrs.dev_spkNeuronIds;
  dev_nPostNeurons = devPtrs.dev_nPostNeurons;
  dev_sparseConVec = devPtrs.dev_sparseConVec;
  dev_sparseIdx = devPtrs.dev_sparseIdx;
  dev_gE[mNeuron] = 0;
  dev_gI[mNeuron] = 0;
  if(mNeuron < N_NEURONS) {
    v[0] = (-1 * 70) +  (40 * randkernel(dev_state)); /* Vm(0) ~ U(-70, -30)*/
    v[1] = 0.3176;
    v[2] = 0.1;
    v[3] = 0.5961;
    localLastNSteps = nstep - STORE_LAST_N_STEPS;
    /* TIMELOOP */
    xx = x1;  
    x = x1;
    h = DT; /*(x2 - x1) / nstep;*/
    isynapNew = 0;
    for (k = 0; k < nstep; k++) {
      dev_IF_SPK[mNeuron] = 0;
      IF_SPK = 0;
      vmOld = v[0];
      derivs(x, v, dv, isynapNew, ibg, iff);
        rk4(v, dv, N_STATEVARS, x, h, vout, isynapNew, ibg, iff);
        /*  ERROR HANDLE     if ((float)(x+h) == x) //nrerror("Step size too small in routine rkdumb");*/
        x += h; 
        xx = x; /*xx[k+1] = x;*/
        /* RENAME */
        for (i = 0; i < N_STATEVARS; ++i) {
          v[i]=vout[i];
        }
        if(k > localLastNSteps) {
          y[mNeuron + N_NEURONS * (k - localLastNSteps)] = v[0];
          dev_isynap[mNeuron + N_NEURONS *  (k - localLastNSteps)] = isynapNew;
        }
        if(k > 2) {
          if(v[0] > SPK_THRESH) { 
            if(vmOld <= SPK_THRESH) {
              IF_SPK = 1;
              dev_IF_SPK[mNeuron] = 1;
              atomicAdd(totNSpks, 1); /* atomic add on global introduces memory latency*/
              localTotNspks = *totNSpks;
	      if(localTotNspks < MAX_SPKS) {
		spkNeuronId[localTotNspks] = mNeuron;
		spkTimes[localTotNspks] = xx - h;
	      }
            }
          }
        }
        __syncthreads(); /* CRUTIAL step to ensure that dev_IF_spk is updated by all threads */
        isynapNew = SparseIsynap(v[0], dev_nPostNeurons, dev_sparseConVec, dev_sparseIdx, IF_SPK);
	/* bg current */
	/*	ibg = bgCur(vmOld); /* make sure AuxRffTotal<<<  >>> is run begore calling bgCur */
	/* FF input current*/
	/*	RffTotal(x);
	Gff(x);
	iff = IFF(vmOld);*/

	__syncthreads();
      }
  }
}
#endif
