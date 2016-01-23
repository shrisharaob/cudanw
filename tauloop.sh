#!/bin/bash
set -eu
set -x
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define TAU_SYNAP_E .*^#define TAU_SYNAP_E ${n}^' devHostConstants.h"    
    nvcc -arch=sm_35 -O9 mysolver.cu -o nw.out
#    ./nw$n.out
done

