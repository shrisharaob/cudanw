#!/bin/bash
set -eu # makes your program exit on error or unbound variable
fldr="/home/dhansel/cuda/cudanw/"
spkfile="spkTimes.csv"
fn="spkTimes_theta" 
extn=".csv"
make

for n in {90..126..18}
do
    echo $n
    ./nw.out 1 $n
    mv $fldr$spkfile $fldr$fn$n$extn
done

