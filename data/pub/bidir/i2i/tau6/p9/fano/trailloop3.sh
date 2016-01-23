#!/bin/bash
# use step 90 and theta start as 0 and 45 , to divide processing on two cards
set -eu # makes your program exit on error or unbound variable
NTRIALS=100
for ((i=75; $i<$NTRIALS; i=$i+1))
do
    for THETA in 0. #22.5 45. 67.5
    do
        echo trial $i theta = $THETA
        ./nw.out 0 0 $THETA $i
    done
done
