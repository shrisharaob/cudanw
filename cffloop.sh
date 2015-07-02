#!/bin/bash
set -eu
set -x
THETA_START=0
THETA_STEP=45
NTRIALS=1
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define CFF_I_PREFACTOR .*^#define CFF_I_PREFACTOR ${n}^' globalVars.h"
    nvcc -arch=sm_35 -O9 mysolver.cu -o nw_k$n.out
done
