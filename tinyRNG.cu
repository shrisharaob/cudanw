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


void ConMatBiDir(float *conVec, int bidirType) {
  /* bidirType: 0  I-I
                1  E-E
  */

  double alpha = 0.0, pBi, pUni, p, k, n;
  unsigned long long i, j;
  tinymt32_t state;
  uint32_t seed = time(0);
  tinymt32_init(&state, seed);
  alpha = (double)ALPHA;
  p = (double)K / (double)NE;
  pBi = alpha * p + (1 - alpha) * p * p;
  pUni = 2 * (1 - alpha) * p * (1 - p);
  printf("\n alpha = %f \n", alpha);
  if(bidirType) {
    printf("\n bidir in E --> E\n");
  }
  else {
    printf("\n bidir in I --> I\n");
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
  }
}

#endif