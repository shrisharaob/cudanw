#ifndef _NEURON_COUNTS
#define _NEURON_COUNTS

#include "hostConstants.h"


/* #define NE 4ULL */
/* #define NI 900ULL */
/* #define N_Neurons (NE + NI) */
/* #define N_NEURONS N_Neurons */
/* #define K 1000.0 */
/* #define DT 0.05 /\* ms*\/ /\* CHANGE EXP+SUM WHEN DT CHANGES   *\/ */
/* #define TAU_SYNAP 3.0 */
/* #define EXP_SUM exp(-1 * DT /TAU_SYNAP)  /\*exp(-1 * DT / TAU_SYNAP !!!!!!!!!!! GENERALIZE !!!!!!! *\/ */
/* #define MAX_UNI_RANDOM_VEC_LENGTH 10000000 //make constant 1e7 */
/* #define STORE_LAST_T_MILLISEC 500.0 */
/* #define STORE_LAST_N_STEPS (STORE_LAST_T_MILLISEC / DT) */
/* #define HOST_CONTRAST 100.0 */
/* #define HOST_CFF 0.100000000 // KFF = CFF * K */


/* #define N_NEURONS_TO_STORE_START 9000 */
/* #define N_NEURONS_TO_STORE_END 9001 */
/* #define N_NEURONS_TO_STORE (N_NEURONS_TO_STORE_END - N_NEURONS_TO_STORE_START) */
/* #define N_E_2BLOCK_NA_CURRENT 1 // number of first n neurons to have their Na2+ currents blocked */
/* #define N_I_2BLOCK_NA_CURRENT 1 */
/* #define N_I_SAVE_CUR 4 */

__constant__ double CONTRAST = HOST_CONTRAST;
__constant__ double theta; // the value of input theta is copied to this symbol, in radians

/* #define ALPHA 0.0 */

/* /\* params patch *\/ */
/* #define L 1.0 */
/* #define CON_SIGMA (L / 1.0) */
/* #define PI 3.14159265359 */

/* /\* feed forward patch parameters *\/ */
/* #define CFF HOST_CFF */
/* #define L_FF 1.0 */
/* #define FF_CON_SIGMA (L_FF / 5.0) */
/* #define NFF 900ULL */
#endif
