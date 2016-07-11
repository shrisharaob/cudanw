#!/bin/bash
# set -eu
# set -x

baseFldr="/homecentral/srao/cuda/data/pub/N4E4/"
for tau_s in $@
do
    for p in 9 95 #8 85 #7 75 8 85 9 95
    do
	cd ${baseFldr}tau${tau_s}/p${p}	
	pwd
#	 rm *.dat
#	./genconvec.out 3 0
	./nw.out 0 0 0 0
    done
done
