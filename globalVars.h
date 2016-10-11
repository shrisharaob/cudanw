#ifndef _GLOBALVARS_
#define _GLOBALVARS_
#include "mycurand.h"
#include "devHostConstants.h"
#include "math.h"

#define MAX_SPKS 50000000
#define PI 3.14159265359
#define SQRT_DT sqrt(DT)
#define DISCARDTIME 2000 // ms, discard time to compute the firing rate

#define Cm 1.0 /* microF / cm^2  */
#define E_Na 55.0 /* mV */
#define E_K -80.0 /*-90*/
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
/*#define L 1.0
#define CON_SIGMA (L / 5.0)
*/

#define CONDUCTANCE_GLOBAL_PREFACTOR 1.0

/* params synapseb */
#define INV_TAU_SYNAP (1 / TAU_SYNAP)
#define V_E 0.0
#define V_I -80.0
#define G_EE (0.15 * CONDUCTANCE_GLOBAL_PREFACTOR)
#define G_EI (2.00 * CONDUCTANCE_GLOBAL_PREFACTOR)
#define G_IE (0.45 * CONDUCTANCE_GLOBAL_PREFACTOR)
#define G_II (3.00 * CONDUCTANCE_GLOBAL_PREFACTOR)

/* backgrund input */
#define RB_E 0.002
#define RB_I 0.002
#define TAU_BG 3.0
#define INV_TAU_BG (1.0 / TAU_BG)
#define G_EB ((0.3 * CONDUCTANCE_GLOBAL_PREFACTOR) / sqrt(K))
#define G_IB ((0.4 * CONDUCTANCE_GLOBAL_PREFACTOR) / sqrt(K))

/* ff input */
//#define CFF 0.1000000000
//#define KFF 200.0
//#define GE_FF (0.95 * 1.0)
//#define GI_FF (1.26 * 1.0)
#define R0 0.02 //0.002
#define R1 0.02 //0.02
#define INP_FREQ (0.004 * PI)
#define ETA_E 1.2
#define ETA_I 1.2
#define MU_E 0.0
#define MU_I 0.0
#define GFF_E ((0.95 * CONDUCTANCE_GLOBAL_PREFACTOR) / sqrt(K))
#define GFF_I ((1.26 * CONDUCTANCE_GLOBAL_PREFACTOR) / sqrt(K))

__device__ float randnXiA[N_Neurons], 
                 randuDelta[N_Neurons], 
                 randwZiA[N_Neurons * 4], 
                 randuPhi[N_Neurons * 3]; 

/* VARIABLES FOR THE FEED FORWARD POISSION ELEMENTS FOR THE CASE OF ORI MAP */
__device__ float POInOriMap[NFF]; // preferred orientation assigned to the layer four neurons
__device__ int IF_SPIKE_POISSION_SPK[NFF];

__device__ double dev_v[N_NEURONS], dev_n[N_NEURONS], dev_z[N_NEURONS], dev_h[N_NEURONS]; 
__device__ double gffItgrl[N_NEURONS];
__device__ double dt, *thetaVec;
__device__ double dev_gE[N_NEURONS], dev_gI[N_NEURONS], dev_isynap[N_NEURONS], dev_iffCurrent[N_Neurons];
__device__ int dev_IF_SPK[N_NEURONS], curConter = 0;


#define SAVE_CURRENT_FOR_NEURON 2
#define N_CURRENT_STEPS_TO_STORE 5


__device__ double dev_GFFmean[N_NEURONS];
__device__ unsigned long long devGFFCounter;



__device__ double glbCurE[N_CURRENT_STEPS_TO_STORE], glbCurI[N_CURRENT_STEPS_TO_STORE]; /* !!!!!! GENERALIZE */
__device__ double rTotal[N_Neurons], gFF[N_Neurons]; /* rTotalPrev[N_Neurons];*/
__device__ double gaussNoiseE[NE], gaussNoiseI[NI];
__device__ curandState bgCurNoiseGenState[N_NEURONS], iffNormRandState[N_NEURONS];
__device__ double dev_bgCur[N_CURRENT_STEPS_TO_STORE], dev_iff[N_CURRENT_STEPS_TO_STORE];
__device__ int dev_nPostNeurons[N_NEURONS], dev_sparseIdx[N_NEURONS]; 

//__device__ int dev_sparseConVec[N_NEURONS * 2 * (int)K + N_NEURONS];

__device__ int dev_nPostNeuronsFF[NFF], dev_sparseIdxFF[NFF]; 

//__device__ int dev_sparseConVecFF[N_NEURONS * (2ULL + (unsigned long long)(CFF * K) + NFF)];

__device__ int *dev_sparseConVec, *dev_sparseConVecFF;

#define N_SPKS_IN_PREV_STEP 3000
__device__ int dev_prevStepSpkIdx[N_NEURONS], /*this will hold the row id in the matrix dev_spksCountMat*/
  dev_ESpkCountMat[N_NEURONS * N_SPKS_IN_PREV_STEP],
  dev_ISpkCountMat[N_NEURONS * N_SPKS_IN_PREV_STEP];



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

#define IF_ORI_MAP 1

typedef struct 
{
  int neuronId, nPostNeurons, *postNeuronIds;
} sparseMat;

typedef struct {
  int *dev_conVec, *dev_nSpks, *dev_spkNeuronIds, k, *dev_sparseConVec, *dev_sparseConVecFF;
  double *dev_vm, *synapticCurrent, *dev_spkTimes, *dev_time;
  curandState *devStates;
} devPtr_t;

typedef struct {
  double tStart, tStop;
  int nSteps;
} kernelParams_t;



struct test_xform {
  __host__ __device__
  void operator() (int* input, int i, int* res_idx, int* res, int nres) const {
    *res_idx++ = input[i];
    *res++ = 1;
  }
 };

// Sum-functor to be used for reduction - just a normal sum of two integers
struct test_sumfun {
  __device__ __host__ int operator() (int res1, int res2) const{
    return res1 + res2;
  }
};

#endif
