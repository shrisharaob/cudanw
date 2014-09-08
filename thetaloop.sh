#!/bin/bash
set -eu # makes your program exit on error or unbound variable
fldr="/homecentral/srao/Documents/code/cuda/cudanw/"
spkfile="spkTimes.csv"
fn="spkTimes_theta" 
extn=".csv"
DEVICE_ID=0
IF_SAVE=0
#make
#./genconvec.out
THETA_START=0
THETA_STEP=45

if [ $# == 2 ]
then
    THETA_START=$1
    THETA_STEP=$2
fi  


for (( THETA = $THETA_START; $THETA<=360; THETA=$THETA+${THETA_STEP} ))
do
    echo $THETA
    if [ $THETA == 315 ]; then
	    ./nw.out $DEVICE_ID 1 $THETA
    else
	    ./nw.out $DEVICE_ID $IF_SAVE $THETA	
    fi
    mv $fldr$spkfile $fldr$fn$THETA$extn
done

