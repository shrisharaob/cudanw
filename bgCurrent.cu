__device__ double bgCur(double vm) {
  unsigned int mNeuron = threadIdx.x + blockIdx.x * blockDim.x;
  //double D = 1;
  double iBg = 0.0;
  double gE, gI, gNoise;
  if(mNeuron < N_NEURONS) {
    if(mNeuron < NE) {
      gNoise = gaussNoiseE[mNeuron];
      gNoise = gNoise * (1 - DT * INV_TAU_BG) + SQRT_DT  * INV_TAU_BG * normRndKernel(bgCurNoiseGenState);
      //      gNoise = 0;
      gE = G_EB * K * K_REC_E_PREFACTOR * (RB_E + sqrt(RB_E / (K * K_REC_E_PREFACTOR)) * gNoise);
      /*      iBg = -1 * gE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));*/
      iBg = -1 * gE * (RHO * vm + (1 - RHO) * E_L);
      gaussNoiseE[mNeuron] = gNoise;
    }
    if(mNeuron >= NE) {
      gNoise = gaussNoiseI[mNeuron - NE];
      gNoise = gNoise * (1 - DT * INV_TAU_BG) +  SQRT_DT  * INV_TAU_BG * normRndKernel(bgCurNoiseGenState);
      //gNoise = 0;
      gI = G_IB * K * K_REC_I_PREFACTOR * (RB_I + sqrt(RB_I / (K * K_REC_I_PREFACTOR)) * gNoise);
      //
      //gI = (sqrt(K * K_REC_I_PREFACTOR) * G_IB * RB_I) + (sqrt(K_REC_I_PREFACTOR) * G_IB * sqrt(RB_I) * gNoise);
      iBg = -1 * gI * (RHO * vm + (1 - RHO) * E_L);
      gaussNoiseI[mNeuron - NE] = gNoise;
    }
    if(mNeuron == SAVE_CURRENT_FOR_NEURON) {
      dev_bgCur[curConter - 1] = iBg;
      }
  }
  //  D +=1;
  return iBg;
}
