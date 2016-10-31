#!/bin/bash
set -eu # makes your program exit on error or unbound variable
fldr="/homecentral/srao/Documents/code/cuda/cudanw/"
DEVICE_ID=0
IF_SAVE=0
FILETAG=0

for THETA in 135.0 157.0 #101.25 123.75 146.25 168.75 #90.0 112.5 135.0 157.0
do
    echo $THETA
    ./nw.out $DEVICE_ID $IF_SAVE $THETA $FILETAG
done

