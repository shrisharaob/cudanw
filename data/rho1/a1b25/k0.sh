#!/bin/bash
set -eu
set -x
THETA_START=0
THETA_STEP=45
NTRIALS=1
    for ((i=0; $i<$NTRIALS; i=$i+1))
    do
    	for THETA in 90.0 112.5 135.0 157.5
    	do
            echo trial $i theta = $THETA
            ./nw.out 0 0 $THETA 0
    	done
    done

