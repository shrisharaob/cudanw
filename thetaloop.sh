#!/bin/bash
set -eu # makes your program exit on error or unbound variable
fldr="/home/dhansel/cuda/cudanw/"
spkfile="spkTimes.csv"
fn="spkTimes_theta" 
extn=".csv"
DEVICE_ID=0
IF_SAVE=0
#make
#./genconvec.out
for THETA in {45..360..45}
do
    echo $THETA
    if [ $THETA == 315 ]; then
	./nw.out $DEVICE_ID $IF_SAVE $THETA
    else
	./nw.out $DEVICE_ID 1 $THETA	
    fi
    mv $fldr$spkfile $fldr$fn$THETA$extn
done

