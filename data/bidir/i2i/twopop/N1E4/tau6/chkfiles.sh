#!/bin/bash

baseFldr="/homecentral/srao/cuda/data/bidir/i2i/twopop/N1E4/"
for p in {0..9}; do
cd p${p}
curdir=$(pwd)
echo " In DIR: " $curdir 
spkfile=$(ls spk*.csv 2> /dev/null)
if [[ -f $spkfile ]]; then
echo "not launching"
else
echo "  LAUNCHING!!!!!"
fi
echo "------------------------------------------------------------------------"
cd ${baseFldr}tau${1}
done
