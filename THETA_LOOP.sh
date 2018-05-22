#!/bin/bash
set -eu
#set -x
FILETAG=0

make clean
make


for PHI in 0 22.5 45 67.5; # 90 112.5 135 157.5;
do
    if [ $PHI -eq 0 ]
    then
	./genconvec.out 3 3
	./nw.out 0 0 $PHI $FILETAG		
    else
        ./nw.out 0 0 $PHI $FILETAG
    fi
done
