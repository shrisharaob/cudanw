#!/bin/bash
for tau_s in $1;
do
    for p in 75 85 95;
    do
	cd /homecentral/srao/cuda/data/pub/bidir/i2i/tau${tau_s}/p${p}
	pwd
	# rm *.dat
	# ./genconvec.out 3 0
	./nw.out 0 0 0 0
    done
done
