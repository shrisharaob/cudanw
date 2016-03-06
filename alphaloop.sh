#!/bin/bash
set -eu
set -x
counter=0
tau=48
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define ALPHA .*^#define ALPHA ${n}^' devHostConstants.h"    
    make clean
    nvcc -arch=sm_35 -O4 mysolver.cu -o nw.out
    mv nw.out /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau}/p${counter}
    nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec.out
    mv genconvec.out /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau}/p${counter}
#    mkdir /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau}/p${counter}/fano


    counter=$((counter+1))     
done
