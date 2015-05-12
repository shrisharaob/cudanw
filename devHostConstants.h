#ifndef _NEURON_COUNTS
#define _NEURON_COUNTS

#define NE 2ULL
#define NI 2ULL
#define N_Neurons (NE + NI)
#define N_NEURONS N_Neurons
#define K 1.0
#define DT 0.05 /* ms*/ /* CHANGE EXP+SUM WHEN DT CHANGES   */
#define TAU_SYNAP 3.0
#define EXP_SUM exp(-1 * DT /TAU_SYNAP)  /*exp(-1 * DT / TAU_SYNAP !!!!!!!!!!! GENERALIZE !!!!!!! */
#define MAX_UNI_RANDOM_VEC_LENGTH 10000000 //make constant 1e7
#define STORE_LAST_T_MILLISEC 2000.0
#define STORE_LAST_N_STEPS (STORE_LAST_T_MILLISEC / DT)
#define HOST_CONTRAST 100.0

#define N_NEURONS_TO_STORE_START 10001
#define N_NEURONS_TO_STORE_END 10010
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
