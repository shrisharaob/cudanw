#!/bin/bash
# use step 90 and theta start as 0 and 45 , to divide processing on two cards
set -eu # makes your program exit on error or unbound variable
NTRIALS=$1
for ((i=0; $i<$NTRIALS; i=$i+1))
do
    for THETA in 101.25 123.75 146.25 168.75
    do
        echo trial $i theta = $THETA
        ./nw.out 0 0 $THETA $i
    done
done
