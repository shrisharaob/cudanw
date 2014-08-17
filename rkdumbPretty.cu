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
      dev_v[mNeuron] = -60;
      dev_n[mNeuron] = 0.3176;
      dev_z[mNeuron] = 0.1;
      dev_h[mNeuron] = 0.5961;
      dev_isynap[mNeuron] = 0;
      dev_gE[mNeuron] = 0;
      dev_gI[mNeuron] = 0;
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
    /* runge kutta 4 */
    derivs(x, v, dv, isynapNew, ibg, iff);
    rk4(v, dv, N_STATEVARS, x, DT, vout, isynapNew, ibg, iff);
    x += DT; 
    /* UPDATE */
    dev_v[mNeuron] = vout[0];
    dev_n[mNeuron] = vout[1];
    dev_z[mNeuron] = vout[2];
    dev_h[mNeuron] = vout[3];
    if(k >= localLastNSteps) {
      y[mNeuron + N_NEURONS * (k - localLastNSteps)] = vout[0];
      synapticCurrent[mNeuron + N_NEURONS *  (k - localLastNSteps)] = isynapNew;
      if(mNeuron == 0) {
	dev_time[k - localLastNSteps] = x;
      }
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

	/*	__syncthreads();  CRUTIAL step to ensure that dev_IF_spk is updated by all threads */
	/*        isynapNew = SparseIsynap(v[0], dev_nPostNeurons, dev_sparseConVec, dev_sparseIdx, IF_SPK);*/
	/* bg current */
	/*	ibg = bgCur(vmOld); /* make sure AuxRffTotal<<<  >>> is run begore calling bgCur */
	/* FF input current*/
	/*	RffTotal(x);
	Gff(x);
	iff = IFF(vmOld);*/
  }
}
#endif
