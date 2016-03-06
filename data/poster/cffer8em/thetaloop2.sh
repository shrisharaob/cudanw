#!/bin/bash
# use step 90 and theta start as 0 and 45 , to divide processing on two cards
set -eu # makes your program exit on error or unbound variable
fldr="/homecentral/srao/Documents/code/cuda/cudanw/"
spkfile="spkTimes.csv"
fn="spkTimes_theta" 
extn=".csv"
DEVICE_ID=0
IF_SAVE=0
#./genconvec.out
THETA_START=0
THETA_STEP=22.5
THETA_END=180
if [ $# == 2 ]
then
    THETA_START=$1
    THETA_STEP=$2
fi  
for THETA in 90. 112.5
do
    echo $THETA
    ./nw.out $DEVICE_ID $IF_SAVE $THETA 0
done

