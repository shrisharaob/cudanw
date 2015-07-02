#!/bin/bash
set -eu
set -x
THETA_START=0
THETA_STEP=45
NTRIALS=1
for n in $(< $1)
do
    echo $n
    for ((i=0; $i<$NTRIALS; i=$i+1))
    do
    	for THETA in 0. 22.5 45.0 67.5 90.0 112.5 135.0 157.5
    	do
            echo trial $i theta = $THETA
            ./nw_k$n.out 0 0 $THETA $n
    	done
    done
done
