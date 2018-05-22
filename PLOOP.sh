#!/bin/bash
set -eu
#set -x
counter=0
counterK=0
p=(0)
tau=(3)
KLIST=(500)
FILETAG=0
RECIP_TYPE='e2i'
#eval sed "-i 's^#define TAU_SYNAP_E .*^#define TAU_SYNAP_E ${tau}^' devHostConstants.h"
for K in 500.0;
do
    for alpha in 0.0;
    do
	cd /homecentral/srao/cuda
	echo "K = " $K "| p = " ${p[counter]} 
	eval sed "-i 's^#define K .*^#define K ${K}^' devHostConstants.h"
        eval sed "-i 's^#define ALPHA .*^#define ALPHA ${alpha}^' devHostConstants.h"	
	make clean
	make
	echo /homecentral/srao/cuda/data/pub/bidir/${RECIP_TYPE}/tau3/p${p[counter]}/K${KLIST[counterK]}
	mkdir -p /homecentral/srao/cuda/data/pub/bidir/${RECIP_TYPE}/tau3/p${p[counter]}/K${KLIST[counterK]}
	mv nw.out /homecentral/srao/cuda/data/pub/bidir/${RECIP_TYPE}/tau3/p${p[counter]}/K${KLIST[counterK]}
        mv genconvec.out /homecentral/srao/cuda/data/pub/bidir/${RECIP_TYPE}/tau3/p${p[counter]}/K${KLIST[counterK]}

	cd /homecentral/srao/cuda/data/pub/bidir/${RECIP_TYPE}/tau3/p${p[counter]}/K${KLIST[counterK]}

	for PHI in 0 22.5 45 67.5 90 112.5 135 157.5;
	do
	    if [ $PHI -eq 0 ]
	    then
		if [ ${p[counter]} -eq 0 ]
		then
		    ./genconvec.out 0 0 0
		else
		    ./genconvec.out 3 3
		fi
		./nw.out 0 0 $PHI $FILETAG		
	    else
         	./nw.out 0 0 $PHI $FILETAG
	    fi
	done
	counter=$((counter+1))
    done
    counter=0
    counterK=$((counterK+1))     	
done

