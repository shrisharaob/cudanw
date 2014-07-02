#ifndef _IFF_
#define _IFF_
// ff input 
_device_ void AuxRffTotal() {
  // draws random numbers from the specified distribution
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  int lNeuron, i;
  long idem1, idem2, idem3, idem4; // seeds for rand generator
  idem1 = -1 * rand();
  idem2 = -1 * rand();
  idem3 = -1 * rand();
  idem4 = -1 * rand();
  for(lNeuron = 1; lNeuron <= N_Neurons; ++lNeuron) {
    randnXiA[lNeuron] =  gasdev(&idem1);
    randuDelta[lNeuron] = PI * ran2(&idem2);
    for(i = 1; i <= 4; ++i) {
      randwZiA[lNeuron][i] = 1.4142135 * sqrt(-1 * log(ran2(&idem3)));
    }
    for(i = 1; i <= 3; ++i) {
      randuPhi[lNeuron][i] = 2 * PI * ran2(&idem4);
    }
  }
}

_device_ void RffTotal(double theta, double t) {
  double etaE, etaI;
  int lNeuron;
  int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  if(mNeuron < N) XF
     for(lNeuron = 1; lNeuron <= NE; ++lNeuron) {
      rTotalPrev[lNeuron] = rTotal[lNeuron]; // rTotal(t - 1)
      rTotal[lNeuron] = CFF * K * (R0 + R1 * log10(1 + contrast)) 
        + sqrt(CFF * K) * R0 * randnXiA[lNeuron]
        + sqrt(CFF * K) * R1 * log10(1 + contrast) * (randnXiA[lNeuron] 
                                                      + etaE * randwZiA[lNeuron][1] * cos(2 * (theta - randuDelta[lNeuron])) 
                                                      + muE * randwZiA[lNeuron][2] * cos(INP_FREQ * t - randuPhi[lNeuron][1])
                                                      + etaE * muE * 0.5 * (randwZiA[lNeuron][3] 
                                                      * cos(2 * theta + INP_FREQ * t - randuPhi[lNeuron][2])
                                                      + randwZiA[lNeuron][4] 
                                                      * cos(2 * theta - INP_FREQ * t + randuPhi[lNeuron][3])));
    }
    for(lNeuron = 1 + NE; lNeuron <= N_Neurons; ++lNeuron) {
      rTotalPrev[lNeuron] = rTotal[lNeuron]; // rTotal(t - 1)
      rTotal[lNeuron] = CFF * K * (R0 + R1 * log10(1 + contrast)) 
        + sqrt(CFF * K) * R0 * randnXiA[lNeuron]
        + sqrt(CFF * K) * R1 * log10(1 + contrast) * (randnXiA[lNeuron] 
                                                      + etaI * randwZiA[lNeuron][1] * cos(2 * (theta - randuDelta[lNeuron])) 
                                                      + muI * randwZiA[lNeuron][2] * cos(INP_FREQ * t - randuPhi[lNeuron][1])
                                                      + etaI * muI * 0.5 * (randwZiA[lNeuron][3] 
                                                      * cos(2 * theta + INP_FREQ * t - randuPhi[lNeuron][2])
                                                      + randwZiA[lNeuron][4] 
                                                      * cos(2 * theta - INP_FREQ * t + randuPhi[lNeuron][3])));

    }}


_device_ void Gff(double theta, double t) {
  int kNeuron;
  double tempGasdev;
  long idem;
  idem = -1 * rand();
  if(t > DT) {
    for(kNeuron = 1; kNeuron <= NE; ++kNeuron) {
      ItgrlOld[kNeuron] = Itgrl[kNeuron];
      tempGasdev = gasdev(&idem);
      Itgrl[kNeuron] = rTotal[kNeuron] + sqrt(rTotal[kNeuron]) * tempGasdev; // / sqrt(dt)
      //      Itgrl[kNeuron] = cos(2 * PI * t) + 10 + sqrt(cos(2 * PI * t) + 10) * gasdev(&idem);
      //gFF[kNeuron] += DT * (- 0.5*(1/1000) * ( (1/K) * gFF[kNeuron] - Itgrl[kNeuron]));
      gFF[kNeuron] += DT * (-1 * GFF_E * sqrt(1/K) * INV_TAU_SYNAP 
                            * ( INV_TAU_SYNAP * gFF[kNeuron] - Itgrl[kNeuron]));
      fprintf(rTotalFP, "%f %f ", gFF[kNeuron], Itgrl[kNeuron]);
    }
    for(kNeuron = NE + 1; kNeuron <= N_Neurons; ++kNeuron) {
      ItgrlOld[kNeuron] = Itgrl[kNeuron];
      Itgrl[kNeuron] = rTotal[kNeuron] + sqrt(rTotal[kNeuron]) * gasdev(&idem); // / SQRT_DT;
      gFF[kNeuron] += DT * (-1 * GFF_I * sqrt(1/K) * INV_TAU_SYNAP 
                            * ( INV_TAU_SYNAP * gFF[kNeuron] - Itgrl[kNeuron]));
      fprintf(rTotalFP, "%f %f ", gFF[kNeuron], rTotal[kNeuron]);
    }
    fprintf(rTotalFP, "\n");
  }
  else {
     for(kNeuron = 1; kNeuron <= NE; ++kNeuron) {
       Itgrl[kNeuron] = rTotal[kNeuron] + sqrt(rTotal[kNeuron]) * gasdev(&idem); // / SQRT_DT;
     }  
     for(kNeuron = NE + 1; kNeuron <= N_Neurons; ++kNeuron) {
       Itgrl[kNeuron] = rTotal[kNeuron] + sqrt(rTotal[kNeuron]) * gasdev(&idem); // / SQRT_DT;
     }
  }
}

_device_ void IFF(double *vm) {
  int mNeuron;
  for (mNeuron = 1; mNeuron <= N_Neurons; ++mNeuron) {
    iFF[mNeuron] = -1 * gFF[mNeuron] * (RHO * (vm[mNeuron] - V_E) + (1 - RHO) * (E_L - V_E));
  }
}
#endif
