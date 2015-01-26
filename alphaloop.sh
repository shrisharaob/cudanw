#!/bin/bash
set -eu
set -x
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define ALPHA .*^#define ALPHA ${n}^' devHostConstants.h"    
#    make clean
#    rm *.dat
#    nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec_t3_a$n.out
    nvcc -arch=sm_35 -O9 mysolver.cu -o nw_t3_a$n.out
#    mv nw_t3_$n.out ../tmp/alpha$n/
#    mv genconvec_t3_$n ../tmp/alpha$n/
#    ./genconvec$n.out 
#    ./nw$n.out

done
