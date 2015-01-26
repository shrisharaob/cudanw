__device__ double bgCur(double vm) {
  unsigned int mNeuron = threadIdx.x + blockIdx.x * blockDim.x;
  double D = 1, iBg = 0;
  double gE, gI, gNoise;
  if(mNeuron < N_NEURONS) {
    if(mNeuron < NE) {
      gNoise = gaussNoiseE[mNeuron];
      gNoise = gNoise * (1 - DT * INV_TAU_BG) + SQRT_DT  * INV_TAU_BG * normRndKernel(bgCurNoiseGenState);
      /*gNoise = 0;*/
      gE = G_EB * K * (RB_E + sqrt(RB_E / K) * gNoise);
      /*gE = G_EB * K * RB_E;*/
      /*      iBg = -1 * gE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));*/
      iBg = -1 * gE * (RHO * vm + (1 - RHO) * E_L);
      gaussNoiseE[mNeuron] = gNoise;
    }
    if(mNeuron >= NE) {
      gNoise = gaussNoiseI[mNeuron - NE];
      gNoise = gNoise * (1 - DT * INV_TAU_BG) +  SQRT_DT  * INV_TAU_BG * normRndKernel(bgCurNoiseGenState);
      /*      gNoise = 0;*/
      gI = G_IB * K * (RB_I + sqrt(RB_I / K) * gNoise);
      /*      gI = G_IB * K * RB_I;*/
      iBg = -1 * gI * (RHO * vm + (1 - RHO) * E_L);
      gaussNoiseI[mNeuron - NE] = gNoise;
    }
    /*    if(mNeuron == SAVE_CURRENT_FOR_NEURON) {
      dev_bgCur[curConter - 1] = gNoise;
      }*/
  }
  D +=1;
  return iBg;
}
