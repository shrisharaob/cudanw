//#ifndef GLOBSLVARS_H
//#include "globalVars.h"

//__constant__ float DEV_EXP = 0.9917;
//__constant__ int DEV_N_NEURONS = 3, DEV_NE = 3;

//float *dev_gEI_E, *dev_gEI_I, *dev_conVec, *gEI_E, *gEI_I;
#ifndef _NEURON_COUNTS
#define _NEURON_COUNTS

#define NE 0
#define NI 100
#define N_Neurons (NE+NI)
#define N_NEURONS N_Neurons
#define K 1.0
#define DT 0.025 // ms
#define TAU_SYNAP 3.0  // ms
#define EXP_SUM 0.99170129 //exp(-1 * DT / TAU_SYNAP)
#define MAX_UNI_RANDOM_VEC_LENGTH 10000000 //make constant 1e7
#define STORE_LAST_T_MILLISEC 100.0
#define STORE_LAST_N_STEPS (STORE_LAST_T_MILLISEC / DT)
#endif
