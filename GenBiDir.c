#ifndef __BIDIR__
#define __BIDIR__
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>

void ConMatBiDir(float *conVec) {
  double alpha, pBi, pUni, p, k, n;
  unsigned long long i, j;
  const gsl_rng_type * T;
  gsl_rng * r;
  p = (double)K / (double)NE;
  alpha = 0.3;
  pBi = alpha * p + (1 - alpha) * p * p;
  pUni = 2 * (1 - alpha) * p * (1 - p);

  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set(r, time(NULL));

  for(i = 0; i < N_NEURONS; ++i) {
    for(j = 0; j < i; ++j) {
      conVec[i + j * N_NEURONS] = 0;
      conVec[j + i * N_NEURONS] = 0;
      conVec[i + i * N_NEURONS] = 0; /* diagnol */
     if(pBi > gsl_rng_uniform (r)) {
        conVec[i + j * N_NEURONS] = 1; // i --> j
        conVec[j + i * N_NEURONS] = 1; //  j --> i
      }
      else {
        if(pUni > gsl_rng_uniform (r)) {
          if(gsl_rng_uniform (r) > 0.5) {
            conVec[j + i * N_NEURONS] = 1; // i --> j
          }
          else {
            conVec[i + j * N_NEURONS] = 1; //  j --> i
          }
        }      
      }
    }
  }
  gsl_rng_free (r);
}

#endif
