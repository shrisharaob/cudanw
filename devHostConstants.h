#ifndef _NEURON_COUNTS
#define _NEURON_COUNTS

#define NE 20000ULL
#define NI 20000ULL
#define N_Neurons (NE + NI)
#define N_NEURONS N_Neurons
#define K_REC_I_PREFACTOR 1.0 // beta
#define K_FF_I_PREFACTOR 1.0 // alpha
#define K_FF_EI_PREFACTOR 1.0
#define PREFACTOR_REC_BG 1.0

#define K (2000.0 / (PREFACTOR_REC_BG * PREFACTOR_REC_BG))
#define DT 0.05 /* ms*/ /* CHANGE EXP+SUM WHEN DT CHANGES   */
#define TAU_SYNAP_E 3.0
#define TAU_SYNAP_I 3.0
#define EXP_SUM_E exp(-1 * DT /TAU_SYNAP_E)
#define EXP_SUM_I exp(-1 * DT /TAU_SYNAP_I)
#define MAX_UNI_RANDOM_VEC_LENGTH 10000000 //make constant 1e7
#define STORE_LAST_T_MILLISEC 1000.0
#define STORE_LAST_N_STEPS (STORE_LAST_T_MILLISEC / DT)
#define HOST_CONTRAST 100.0

#define N_NEURONS_TO_STORE_START 0
#define N_NEURONS_TO_STORE_END 1
#define N_NEURONS_TO_STORE (N_NEURONS_TO_STORE_END - N_NEURONS_TO_STORE_START)
#define N_E_2BLOCK_NA_CURRENT 1 // number of first n neurons to have their Na2+ currents blocked
#define N_I_2BLOCK_NA_CURRENT 1
#define N_I_SAVE_CUR 1

__constant__ double CONTRAST = HOST_CONTRAST;
__constant__ double theta;

#define ALPHA 0.0

/* params patch */
#define L 1.0
#define CON_SIGMA (L / 5.0)
#define PI 3.14159265359
#endif
