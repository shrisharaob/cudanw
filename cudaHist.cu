#ifndef _CUDA_HIST_WRAPPER_
#define _CUDA_HIST_WRAPPER_
#include "cuda_histogram.h"
#include <assert.h>
#include <stdio.h>

    // minimal test - 1 key per input index
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