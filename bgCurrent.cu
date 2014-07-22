__device__ float bgCur(float vm) {
  int mNeuron = threadIdx.x + blockIdx.x * blockDim.x;
  double D = 1, iBg = 0;
  double gE, gI, gNoise;
  if(mNeuron < N_NEURONS) {
    if(mNeuron < NE) {
      gNoise = gaussNoiseE[mNeuron];
      gNoise = gNoise * (1 - DT * INV_TAU_SYNAP) + DT * SQRT_DT  * INV_TAU_SYNAP * normRndKernel(bgCurNoiseGenState);
      gE = G_EB * K * (RB_E + sqrt(RB_E / K) * gNoise);
      iBg = -1 * gE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      gaussNoiseE[mNeuron] = gNoise;
    }
    if(mNeuron >= NE) {
      gNoise = gaussNoiseI[mNeuron - NE];
      gNoise = gNoise * (1 - DT * INV_TAU_SYNAP) + DT * SQRT_DT  * INV_TAU_SYNAP * normRndKernel(bgCurNoiseGenState);
      gI = G_IB * K * (RB_I + sqrt(RB_I / K) * gNoise);
      iBg = -1 * gI * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      gaussNoiseI[mNeuron - NE] = gNoise;
    }
    if(mNeuron == 16003) {
      dev_bgCur[curConter - 1] = iBg;
    }
  }
  return iBg;
}
