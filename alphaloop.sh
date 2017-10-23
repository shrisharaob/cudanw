#!/bin/bash
set -eu
set -x
counter=0
tau=$2
# baseFldr="/homecentral/srao/cuda/data/pub/N3E4/"
baseFldr="/homecentral/srao/cuda/data/bidir/i2i/twopop/N1E4/"
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define ALPHA .*^#define ALPHA ${n}^' devHostConstants.h"
    eval sed "-i 's^#define TAU_SYNAP_E .*^#define TAU_SYNAP_E ${2}.0^' devHostConstants.h"    
    make clean
    nvcc -arch=sm_35 -O4 mysolver.cu -o nw.out
    # mkdir -p ${baseFldr}tau${tau}/p${counter}
    mv -v nw.out ${baseFldr}tau${tau}/p${counter}

    # nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec.out

    # mv -v genconvec.out ${baseFldr}tau${tau}/p${counter}

    # cd ${baseFldr}tau${tau}/p${counter}
    # ./genconvec.out
    # cd /homecentral/srao/cuda


    counter=$((counter+1))     
done
    # mv genconvec.out /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau}/p${counter}
    # mv nw.out /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau}/p${counter}
