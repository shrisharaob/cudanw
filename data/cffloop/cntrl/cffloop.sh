#!/bin/bash
set -eu
set -x
THETA_START=0
THETA_STEP=45
NTRIALS=1
    for ((i=0; $i<$NTRIALS; i=$i+1))
    do
    	for THETA in 0. 22.5 45.0 67.5 90.0
    	do
            echo trial $i theta = $THETA
            ./nw.out 0 0 $THETA 0
    	done
    done

