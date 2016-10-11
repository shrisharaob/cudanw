#!/bin/bash
set -eu # makes your program exit on error or unbound variable
fldr="/homecentral/srao/Documents/code/cuda/cudanw/"
DEVICE_ID=0
IF_SAVE=1
FILETAG=vm

for THETA in  11.25 33.75 56.25 78.75  #0.0 22.5 45.0 67.5 #90.0 112.5 135.0 157.0
do
    echo $THETA
    ./nw.out $DEVICE_ID $IF_SAVE $THETA $FILETAG
done

