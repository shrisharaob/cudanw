#ifndef __FIXED_EII__
#define __FIXED_EII__
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>

void ConMatFixedEII(float *conVec) {
  double p, k, n;
  unsigned long long i, j;
  const gsl_rng_type * T;
  gsl_rng * r;
  p = (double)K / (double)NE;
  alpha = 0.3;
  pBi = alpha * p + (1 - alpha) * p * p;
  pUni = 2 * (1 - alpha) * p * (1 - p);
  // setup gsl rng 
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  rFixed = gsl_rng_alloc (T);
  gsl_rng_set(r, time(NULL));
  gsl_rng_set(rFixed, 12345ULL);

  for(i = 0; i < N_NEURONS; ++i) {
    for(j = 0; j < i; ++j) {
      conVec[j + i * N_NEURONS] = 0;
      if(i < NE && j < NE) { // E --> E
	if(p >= gsl_rng_uniform(rFixed)) {
	    conVec[j + i * N_NEURONS] = 1;
	}
      }
      else {
	if(p >= gsl_rng_uniform(r)) {
	  conVec[j + i * N_NEURONS] = 1;
	}
      }
    }
  }
  gsl_rng_free (r);
}

#endif
