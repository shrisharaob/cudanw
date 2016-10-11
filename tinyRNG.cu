#ifndef __TINY_RNG_CU__
#define __TINY_RNG_CU__
#include <stdint.h>
#include "tinyRGN.c"
#include "devHostConstants.h"

// #ifndef __CONCAT
// #define __CONCATenate(left, right) left ## right
// #define __CONCAT(left, right) __CONCATenate(left, right)
// #endif

// #define UINT32_C(value)   __CONCAT(value, UL)

void tsttiny() {
  /* tinymt32_t fixedState;
 uint32_t seedFixed = 12543;
 tinymt32_init(&fixedState, seedFixed);*/

 printf("inside tst\n");
 for(int i = 0; i <= 3; ++i) 
   {
     printf("%f \n", (double)rand() / (double)RAND_MAX );
   //   printf("%f \n", tinymt32_generate_float(&fixedState));
   }
 printf("tst done\n");
}

void ConMatFixedEII(float *conVec) {
  double p;
  unsigned long long i, j;
  tinymt32_t state, fixedState;
  uint32_t seed = time(0);
  uint32_t seedFixed = 12543;
  tinymt32_init(&state, seed);
  tinymt32_init(&fixedState, seedFixed);
  double tmp = 0;
  for(i = 0; i < N_NEURONS; ++i) {
    if(i < NE) {
      p = (double)K / (double)NE;
    }
    else {
      p = (double)K / (double)NI;
    }
    for(j = 0; j < i; ++j) {
      conVec[j + i * N_NEURONS] = 0;
      if(i < NE && j < NE) { // E --> E
	/*	*/
	tmp = tinymt32_generate_float(&state);
	if(p >= tmp) {
	    conVec[j + i * N_NEURONS] = 1;
	}
      }
      else {
	tmp = tinymt32_generate_float(&fixedState);
	tmp = (double)rand() / (double)RAND_MAX ;
	if(p >= tmp) {
	  conVec[j + i * N_NEURONS] = 1;
	}
      }
    }
  }
}

void ConMatBiDir(float *conVec, int bidirType) {
  /* bidirType: 0  I-I
                1  E-E
  */

  double alpha = 0.0, pBi, pUni, p;
  unsigned long long i, j;
  tinymt32_t state;
  uint32_t seed = time(0);
  tinymt32_init(&state, seed);
  alpha = (double)ALPHA;
  p = (double)K / (double)NE;
  pBi = alpha * p + (1 - alpha) * p * p;
  pUni = 2 * (1 - alpha) * p * (1 - p);
  printf("\n alpha = %f \n", alpha);
  if(bidirType == 1) {
    printf("\n bidir in E --> E\n");
  }
  else if(bidirType == 0) {
    printf("\n bidir in I --> I\n");
  }
  else {
    printf("\n bidir in  E --> E and I --> I\n");
  }
  /* INITIALIZE */
  for(i = 0; i < N_NEURONS; ++i) {
    for(j =0; j < N_NEURONS; ++j) {
      conVec[i + j * N_NEURONS] = 0;
    }
  }
  /* COMPUTE p */
  for(i = 0; i < N_NEURONS; ++i) {
    if(i < NE) {
      p = (double)K / (double)NE;
    }
    else {
      p = (double)K / (double)NI;
    }
    pBi = alpha * p + (1 - alpha) * p * p;
    pUni = 2 * (1 - alpha) * p * (1 - p);
    /* BI-DIR CONNECTIONS IN I -->I   */
    if(bidirType == 0) { 
      for(j = 0; j < NE; ++j) { /* E/I --> E */
        if(p >= tinymt32_generate_float(&state)) {
          conVec[i + j * N_NEURONS] = 1;
        }
      }
      if(i < NE){  /* E --> I */
        for(j = NE; j < N_NEURONS; ++j) {
          if(p >= tinymt32_generate_float(&state)) {
            conVec[i + j * N_NEURONS] = 1;
          }
        }
      }
      if(i >= NE) {
        for(j = NE; j < i; ++j) {/* I --> I */
          if(pBi > tinymt32_generate_float(&state)) {
            conVec[i + j * N_NEURONS] = 1; // i --> j
            conVec[j + i * N_NEURONS] = 1; //  j --> i
          }
          else {
            if(pUni > tinymt32_generate_float(&state)) {
              if(tinymt32_generate_float(&state) > 0.5) {
                conVec[j + i * N_NEURONS] = 1; // i --> j
              }
              else {
                conVec[i + j * N_NEURONS] = 1; //  j --> i
              }
            }      
          }
        }
      }
    }
    /* BI-DIR CONNECTIONS IN E --> E   */
    if(bidirType == 1) {
      for(j = NE; j < N_NEURONS; ++j) { /* E/I --> I */
        if(p >= tinymt32_generate_float(&state)) {
          conVec[i + j * N_NEURONS] = 1;
        }
      }
      if(i >= NE) {
        for(j = 0; j < NE; ++j) { /* I --> E */
          if(p >= tinymt32_generate_float(&state)) {
            conVec[i + j * N_NEURONS] = 1;
          }
        }
      }
      if(i < NE) {
        for(j = 0; j < i; ++j) {/* E --> E */
          if(pBi > tinymt32_generate_float(&state)) {
            conVec[i + j * N_NEURONS] = 1; // i --> j
            conVec[j + i * N_NEURONS] = 1; //  j --> i
          }
          else {
            if(pUni > tinymt32_generate_float(&state)) {
              if(tinymt32_generate_float(&state) > 0.5) {
                conVec[j + i * N_NEURONS] = 1; // i --> j
              }
              else {
                conVec[i + j * N_NEURONS] = 1; //  j --> i
              }
            }      
          }
        }
      }
    }

    /* BIDIR IN BOTH E-E AND I-I*/
    if(bidirType == 2) {
      if(i < NE) {
        for(j = 0; j < NE; ++j) { /* E --> E */
          if(pBi > tinymt32_generate_float(&state)) {
            conVec[i + j * N_NEURONS] = 1; // i --> j
            conVec[j + i * N_NEURONS] = 1; //  j --> i
          }
          else {
            if(pUni > tinymt32_generate_float(&state)) {
              if(tinymt32_generate_float(&state) > 0.5) {
                conVec[j + i * N_NEURONS] = 1; // i --> j
              }
              else {
                conVec[i + j * N_NEURONS] = 1; //  j --> i
              }
            }  
          }
        }
      }
      if(i < NE){  /* E --> I */
        for(j = NE; j < N_NEURONS; ++j) {
          if(p >= tinymt32_generate_float(&state)) {
            conVec[i + j * N_NEURONS] = 1;
          }
        }
      }

      if(i >= NE) {
        for(j = 0; j < NE; ++j) { /* I -> E  */
          if(p >= tinymt32_generate_float(&state)) {
            conVec[i + j * N_NEURONS] = 1;
          }
        }
        for(j = NE; j < i; ++j) {/* I --> I */
          if(pBi > tinymt32_generate_float(&state)) {
            conVec[i + j * N_NEURONS] = 1; // i --> j
            conVec[j + i * N_NEURONS] = 1; //  j --> i
          }
          else {
            if(pUni > tinymt32_generate_float(&state)) {
              if(tinymt32_generate_float(&state) > 0.5) {
                conVec[j + i * N_NEURONS] = 1; // i --> j
              }
              else {
                conVec[i + j * N_NEURONS] = 1; //  j --> i
              }
            }      
          }
        }
      }
    }



  }
}

#endif
