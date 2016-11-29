#!/bin/bash
set -eu
set -x
counter=0
tau=30
# baseFldr="/homecentral/srao/cuda/data/pub/N3E4/"
baseFldr="/homecentral/srao/cuda/data/bidir/i2i/twopop/N1E4/"
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define ALPHA .*^#define ALPHA ${n}^' devHostConstants.h"    
    make clean
    nvcc -arch=sm_35 -O4 mysolver.cu -o nw.out
    mkdir -p ${baseFldr}tau${tau}/p${counter}
    mv -v nw.out ${baseFldr}tau${tau}/p${counter}

    nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec.out

    mv -v genconvec.out ${baseFldr}tau${tau}/p${counter}


    counter=$((counter+1))     
done
    # mv genconvec.out /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau}/p${counter}
    # mv nw.out /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau}/p${counter}
