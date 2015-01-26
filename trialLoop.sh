#!/bin/bash
# run trials 
# use step 2 and theta start as 0 and 1 , to divide processing on two cards
set -eu # makes your program exit on error or unbound variable
fldr="/homecentral/srao/Documents/code/cuda/cudanw/"
spkfile="spkTimes.csv"
fn="spkTimes_trial" 
extn=".csv"
DEVICE_ID=0
IF_SAVE=0
#make
#./genconvec.out
TRIAL_ID_START=0
TRIAL_ID_STEP=1

if [ $# == 2 ]
then
    TRIAL_ID_START=$1
    TRIAL_ID_STEP=$2
fi  

#screen -dmS pc$TRIAL_ID_START_

for (( TRIAL = $TRIAL_ID_START; $TRIAL<10; TRIAL=$TRIAL+${TRIAL_ID_STEP} ))
do
    echo $TRIAL
    if [ $TRIAL == 315 ]; then
	    ./nw.out $DEVICE_ID 1 $TRIAL 
    else
	    ./nw.out $DEVICE_ID $IF_SAVE $TRIAL 
    fi
    mv $fldr$spkfile $fldr$fn$TRIAL$extn
done
