#!/bin/bash
set -eu
set -x
counter=0
counterTau=0
p=(75 85 95)
tau=(6 12 24 48)
eval sed "-i 's^#define TAU_SYNAP_E .*^#define TAU_SYNAP_E ${tau}^' devHostConstants.h"

for tt in 6 12 24 48
do
    for n in $(< $1)
    do
#    basefolder="/homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau[counterTau]}/"    
	echo $n
	echo ${p[counter]} $tt
	echo i2i/tau${tau[counterTau]}/p${p[counter]} 
	eval sed "-i 's^#define ALPHA .*^#define ALPHA ${n}^' devHostConstants.h"
        eval sed "-i 's^#define TAU_SYNAP_E .*^#define TAU_SYNAP_E ${tau[counterTau]}.0^' devHostConstants.h"
	make clean
	nvcc -arch=sm_35 -O4 mysolver.cu -o nw.out

#	mkdir -p /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau[counterTau]}/p${p[counter]}

	mv nw.out /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau[counterTau]}/p${p[counter]}

	# nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec.out    
	# mv genconvec.out /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau[counterTau]}/p${p[counter]}
    counter=$((counter+1))
    done
    counter=0
    counterTau=$((counterTau+1))     	
done

