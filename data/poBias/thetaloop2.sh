#!/bin/bash
# use step 90 and theta start as 0 and 45 , to divide processing on two cards
set -eu # makes your program exit on error or unbound variable
fldr="/homecentral/srao/Documents/code/cuda/cudanw/"
spkfile="spkTimes.csv"
fn="spkTimes_theta" 
extn=".csv"
DEVICE_ID=0
IF_SAVE=1

THETA_START=0
THETA_STEP=22.5

if [ $# == 2 ]
then
    THETA_START=$1
    THETA_STEP=$2
fi  

#screen -dmS pc$THETA_START_

for THETA in 90 112.5 135 157.5; #(( THETA = $THETA_START; $THETA<180; THETA=$THETA+${THETA_STEP} ))
do
    echo $THETA
    if [ $THETA == 315 ]; then
    	    ./nw.out $DEVICE_ID 1 $THETA 0
    else
    	    ./nw.out $DEVICE_ID $IF_SAVE $THETA 0
    fi
    #mv $fldr$spkfile $fldr$fn$THETA$extn
done

