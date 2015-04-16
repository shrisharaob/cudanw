#!/bin/bash
set -eu
set -x
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define CONDUCTANCE_GLOBAL_PREFACTOR .*^#define CONDUCTANCE_GLOBAL_PREFACTOR ${n}^' globalVars.h"    
    nvcc -arch=sm_35 -O9 mysolver.cu -o nw_g$n.out
    ./nw_g$n.out
done
