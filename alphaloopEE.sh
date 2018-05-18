#!/bin/bash
# set -eu
set -x
counter=0
counterTau=0
#p=(7 75 8 85 9 95)
#realP=(0.7 0.75 0.8 0.85 0.9 0.95)
#tau=(3 6 12 24 48)
tau=(3)
p=(2)
baseFldr="/homecentral/srao/cuda/data/pub/mOne0/e2e/N1E4/T5/"
for tt in 3; #3 6 12 24 48
do
    for n in 0.2; #0.85 0.9 0.95 #0.7 0.75 0.8 0.85 0.9 0.95
    do
	cd /homecentral/srao/cuda
	echo "-==================-"
	echo $n
	echo ${baseFldr}tau${tau[counterTau]}/p${p[counter]}	
	mkdir -p ${baseFldr}tau${tau[counterTau]}/p${p[counter]}
 	eval sed "-i 's^#define ALPHA .*^#define ALPHA ${n}^' devHostConstants.h"
        eval sed "-i 's^#define TAU_SYNAP_E .*^#define TAU_SYNAP_E ${tau[counterTau]}.0^' devHostConstants.h"

 	make clean
 	nvcc -arch=sm_35 -O4 mysolver.cu -o nw.out
        mv nw.out ${baseFldr}tau${tau[counterTau]}/p${p[counter]}
 	nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec.out    
 	mv genconvec.out ${baseFldr}tau${tau[counterTau]}/p${p[counter]}
	cd ${baseFldr}tau${tau[counterTau]}/p${p[counter]}
	./genconvec.out 3 1
	./nw.out 0 0 0 0 

    counter=$((counter+1))
    done
    counter=0
    counterTau=$((counterTau+1))     	
done

