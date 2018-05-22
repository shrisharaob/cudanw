#!/bin/bash
set -eu
#set -x
FILETAG=0

#sleep 4m

for PHI in 90 112.5 135 157.5;
do
    ./nw.out 0 0 $PHI $FILETAG
done
