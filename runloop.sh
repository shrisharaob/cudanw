#!/bin/bash
set -eu
set -x
counter=0
tau=$2
# baseFldr="/homecentral/srao/cuda/data/pub/N3E4/"
baseFldr="/homecentral/srao/cuda/data/bidir/i2i/twopop/N1E4/"
for n in $(< $1)
do
    echo $n
    cd ${baseFldr}tau${tau}/p${counter}
    at now +${n}min <<EOF
screen -dmS a ./nw.out 0 0 0 0 
EOF
    cd /homecentral/srao/cuda


    counter=$((counter+1))     
done
