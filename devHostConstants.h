#ifndef _NEURON_COUNTS
#define _NEURON_COUNTS

#define NE 40000ULL
#define NI 10000ULL
#define N_Neurons (NE + NI)
#define N_NEURONS N_Neurons
#define K 2000.0
#define DT 0.025 /* ms*/
#define TAU_SYNAP 3.0  /* ms*/
#define EXP_SUM 0.991701292638875 /*0.98347145382161749*/ /**/ /*exp(-1 * DT / TAU_SYNAP) !!!!!!!!!!! GENERALIZE !!!!!!! */
#define MAX_UNI_RANDOM_VEC_LENGTH 10000000 //make constant 1e7
#define STORE_LAST_T_MILLISEC 100.0
#define STORE_LAST_N_STEPS (STORE_LAST_T_MILLISEC / DT)

__constant__ double CONTRAST = 0.0;
__constant__ double theta;
#endif
