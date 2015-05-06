#!/bin/bash
set -eu
set -x
THETA=0.0
THETA_STEP=45
NTRIALS=$1
for n in $(< $2)
do
    echo $n
    eval sed "-i 's^#define HOST_CONTRAST .*^#define HOST_CONTRAST ${n}^' devHostConstants.h"    
    nvcc -arch=sm_35 -O9 mysolver.cu -o nw_alpha$n.out
    for ((i=0; $i<$NTRIALS; i=$i+1))
    do
	for THETA in 0.0 22.5 45. 67.5 90. 112.5 135. 157.5
	do
            echo trial $i theta = $THETA
	    ./nw_alpha$n.out 0 0 $THETA $i
	done
    done
done
