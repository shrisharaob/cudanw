#ifndef _GLOBALVARS_
#define _GLOBALVARS_
#include "mycurand.h"
#include "devHostConstants.h"

#define MAX_SPKS 105000000
#define PI 3.14159265359
#define SQRT_DT sqrt(DT)

#define Cm 1.0 /* microF / cm^2  */
#define E_Na 55.0 /* mV */
#define E_K -90.0
#define E_L -65.0
#define G_Na 100.0 /* mS/cm^2 */
#define G_K 40.0
#define G_L_E 0.05 /* excitatory*/
#define G_L_I  0.1 /*inhibitory*/
#define G_adapt 0.5
#define Tau_adapt 60.0 /* in ms*/

/* params network*/
#define N_STATEVARS 4 /* equals the number of 1st order ode's */

/* params patch */
#define L 1.0
#define CON_SIGMA (L / 5.0)

/* params synapseb */
#define INV_TAU_SYNAP (1 / TAU_SYNAP)
#define V_E  0.0
#define V_I -80.0
#define G_EE 0.15
#define G_EI 2.00
#define G_IE 0.45
#define G_II 3.0 

/*
#define G_EE 0.0
#define G_EI 0.00
#define G_IE 0.0
#define G_II 0.0 
*/

/* backgrund input */
#define RB_E 3.0
#define RB_I 3.0

#define G_EB (0.3 /sqrt(K))
#define G_IB (0.4 /sqrt(K))

/* ff input */
#define CFF 0.1
#define KFF 1.0
#define GE_FF 0.95
#define GI_FF 1.26
#define R0 0.02
#define R1 0.2
#define INP_FREQ (4 * PI)
#define ETA_E 0.4
#define ETA_I 0.4
#define MU_E 0.1
#define MU_I 0.1
#define GFF_E 0.95
#define GFF_I 1.26

__device__ float randnXiA[N_Neurons], 
                 randuDelta[N_Neurons], 
                 randwZiA[N_Neurons * 4], 
                 randuPhi[N_Neurons * 3]; 
__device__ double dev_v[N_NEURONS], dev_n[N_NEURONS], dev_z[N_NEURONS], dev_h[N_NEURONS]; 
__device__ double dt, *thetaVec;
__device__ double dev_gE[N_NEURONS], dev_gI[N_NEURONS], dev_isynap[N_NEURONS];
__device__ int dev_IF_SPK[N_NEURONS], curConter = 0;
__device__ double glbCurE[5 * 4000], glbCurI[5 * 4000]; /* !!!!!! GENERALIZE */
__device__ double rTotal[N_Neurons], gFF[N_Neurons]; /* rTotalPrev[N_Neurons];*/
__device__ float gaussNoiseE[NE], gaussNoiseI[NI];
__device__ curandState bgCurNoiseGenState[N_NEURONS], iffNormRandState[N_NEURONS];
__device__ double dev_bgCur[5 * 4000], dev_iff[5000];
__device__ int dev_nPostNeurons[N_NEURONS], dev_sparseIdx[N_NEURONS]; 
__device__ int dev_sparseConVec[N_NEURONS * 2 * (int)K + N_NEURONS];

/* // recurrent input  */
/* __device__ double *tempCurE, *tempCurI; */
/* //__device__ double *iBg, *gaussNoiseE, *gaussNoiseI; */
/* __device__ double *input_cur, *IF_SPK, conMat[N_NEURONS], nSpks; */
/* __device__ double *iSynap, *expSum;// *gEI_E, *gEI_I; */
/* //__device__ FILE *outVars, *spkTimesFp, *isynapFP, *gbgrndFP, *gEEEIFP, *vmFP; */
/* __device__ double contrast, theta; */
/* __device__ double *gFF, *iFF, *rTotal, muE, muI, */
/*   *randnXiA, // norm rand number */
/*   **randwZiA, // weibul rand number */
/*   *randuDelta, // uniform rand (0, PI) */
/*   **randuPhi, // uniform rand (0, 2.PI) */
/*   *rTotalPrev, //  rToral(t - 1) */
/*   *tempRandnPrev, // randn prev (eq. 15) */
/*   *tempRandnNew, */
/*   *Itgrl, *ItgrlOld; */


#define RHO 0.5 /* ratio - smatic / dendritic synapses*/
#define SPK_THRESH 0.0

typedef struct 
{
  int neuronId, nPostNeurons, *postNeuronIds;
} sparseMat;

typedef struct {
  int *dev_conVec, *dev_nSpks, *dev_spkNeuronIds, k;
  double *dev_vm, *synapticCurrent, *dev_spkTimes, *dev_time;
  curandState *devStates;
} devPtr_t;

typedef struct {
  double tStart, tStop;
  int nSteps;
} kernelParams_t;

#endif
