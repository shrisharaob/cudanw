#include <cuda.h>
#include "globalVars.h"
#include "devFunctionProtos.h"
#include "derivs.cu"
#include "rk4.cu"

__global__ void rkdumb(float *vstart, int nvar, float x1, float x2, int nstep, int *totNSpks, float *spkTimes, int *spkNeuronId, float *y, int *dev_conVec, float *dev_isynap) { 
  //  void rk4(float v[], float dydx[], int n,  float x, float h, float yout[], void (*derivs)(float, float [], float []));
  int i, k;
  float x, h, xx, isynapNew = 0;// isynapOld = 0; //vm
  float v[N_STATEVARS], vout[N_STATEVARS], dv[N_STATEVARS], vmOld;
  int localTotNspks;
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  localTotNspks = *totNSpks;
  __syncthreads();
  if(mNeuron < N_NEURONS) {
  //*** START ***//
  for (i = 0; i < nvar; i++) {
    v[i] = vstart[i + mNeuron * N_STATEVARS];
  }
  //*** TIMELOOP ***//
  xx = x1;  
  x = x1;
  h = DT; //(x2 - x1) / nstep;
  for (k = 0; k < nstep; k++) 
    {
      dev_IF_SPK[mNeuron] = 0;
      vmOld = v[0];
      derivs(x, v, dv, isynapNew);
      rk4(v, dv, N_STATEVARS, x, h, vout, isynapNew);
      //  ERROR HANDLE     if ((float)(x+h) == x) //nrerror("Step size too small in routine rkdumb");
      x += h; 
      xx = x; //xx[k+1] = x;
      /* RENAME */
      for (i = 0; i < nvar; ++i) 
        {
          v[i]=vout[i];
          //  y[i][k+1] = v[i];
        }
      y[mNeuron + N_NEURONS * k] = v[0];
      if(k > 2) {
        //        for(mNeuron = 1; mNeuron <= N_Neurons; ++mNeuron) {
        //          clmNo = (mNeuron - 1) * N_STATEVARS;
          //          IF_SPK[mNeuron] = 0; // GLOBAL 
        //vm = v[0]; // + clmNo];  /// used later for 
          //          spkNeuronId = -1;
        if(v[0] > SPK_THRESH) { 
          if(vmOld <= SPK_THRESH) {
            dev_IF_SPK[mNeuron] = 1;
            spkNeuronId[localTotNspks] = mNeuron;
            spkTimes[localTotNspks] = xx;
            atomicAdd(totNSpks, 1); // atomic add on global introduces memory latency
          }
        }
      }
      __syncthreads(); // so that dev_IF_spk is updated by all threads 
      isynapNew = isynap(v[0], dev_conVec);
      dev_isynap[mNeuron + N_NEURONS * k] = isynapNew;
      //      }
      //      CudaISynap(spkNeuronId); // allocate memore on device for spkNeuronId vector
      //      ISynapCudaAux(vm); // returns current 
      //      IBackGrnd(vm);
      // FF input current
      //      RffTotal(theta, x);
      //      Gff(theta, x);
      //      IFF(vm);
             //      __syncthreads();
    }
  }
}
