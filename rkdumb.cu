#ifndef _RKDUMB_
#define _RKDUMB_
#include <cuda.h>
#include "globalVars.h"
#include "devFunctionProtos.h"
#include "derivs.cu"
#include "rk4.cu"

__global__ void rkdumb(double x1, double x2, int nstep, int *totNSpks, double *spkTimes, int *spkNeuronId, double *y, int *dev_conVec, double *dev_isynap, curandState *dev_state) { 
  int i, k;
  double x, h, xx, isynapNew = 0;// isynapOld = 0; //vm
  double v[N_STATEVARS], vout[N_STATEVARS], dv[N_STATEVARS], vmOld;
  int localTotNspks = 0, localLastNSteps;
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;

  if(mNeuron < N_NEURONS) {
    v[0] = (-1 * 70) +  (40 * randkernel(dev_state)); // Vm(0) ~ U(-70, -30)
    v[1] = 0.3176;
    v[2] = 0.1;
    v[3] = 0.5961;
    localLastNSteps = nstep - STORE_LAST_N_STEPS;
    //*** TIMELOOP ***//
    xx = x1;  
    x = x1;
    h = DT; //(x2 - x1) / nstep;
    isynapNew = 0;
    for (k = 0; k < nstep; k++) 
      {
        dev_IF_SPK[mNeuron] = 0;
        vmOld = v[0];
        derivs(x, v, dv, isynapNew);
        rk4(v, dv, N_STATEVARS, x, h, vout, isynapNew);
        //  ERROR HANDLE     if ((double)(x+h) == x) //nrerror("Step size too small in routine rkdumb");
        x += h; 
        xx = x; //xx[k+1] = x;
        /* RENAME */
        for (i = 0; i < N_STATEVARS; ++i) {
          v[i]=vout[i];
        }
        if(k > localLastNSteps) {
          y[mNeuron + N_NEURONS * (k - localLastNSteps)] = v[0];
          dev_isynap[mNeuron + N_NEURONS * (k - localLastNSteps)] = isynapNew;
        }
        if(k > 2) {
          if(v[0] > SPK_THRESH) { 
            if(vmOld <= SPK_THRESH) {
	      dev_IF_SPK[mNeuron] = 1;
              atomicAdd(totNSpks, 1); /* atomic add on global introduces memory latency */
              localTotNspks = *totNSpks;
	      if(localTotNspks < MAX_SPKS) {
		spkNeuronId[localTotNspks] = mNeuron;
		spkTimes[localTotNspks] = xx;
	      }
	    }
          }
        }
        __syncthreads(); /* CRUTIAL step to ensure that dev_IF_spk is updated by all threads */
        isynapNew = isynap(v[0], dev_conVec);

	/*
        // if(k == (nstep - STORE_LAST_N_STEPS) || (nstep - STORE_LAST_N_STEPS) < 0) {
       //   y[mNeuron + N_NEURONS * k] = v[0];
        //   dev_isynap[mNeuron + N_NEURONS * k] = isynapNew;
        // } 



        //      }
        //      CudaISynap(spkNeuronId); // allocate memore on device for spkNeuronId vector
        //      ISynapCudaAux(vm); // returns current 
        //      IBackGrnd(vm);
        // FF input current
        //      RffTotal(theta, x);
        //      Gff(theta, x);
        //      IFF(vm);
        //      __syncthreads();
	
	*/

      }
  }
}
#endif
