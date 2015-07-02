#!/bin/bash
set -eu
set -x
THETA_START=0
THETA_STEP=45
NTRIALS=1
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define K_I_REC_FF_RATIO .*^#define K_I_REC_FF_RATIO ${n}^' devHostConstants.h"
    nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec_k$n.out
    nvcc -arch=sm_35 -O9 mysolver.cu -o nw_k$n.out
done
