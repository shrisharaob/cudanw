#!/bin/bash

baseFldr="/homecentral/srao/cuda/data/bidir/i2i/twopop/N1E4/"
for p in {9}; do
    cd ${baseFldr}tau${1}
    cd p${p}
    curdir=$(pwd)
    echo " In DIR: " $curdir
    spkName=spkTimes_xi0.8_theta0_0.${p}0_${1}.0_cntrst100.0_100000_tr0.csv
    echo $spkName
    spkfile=$(ls ${spkName} 2> /dev/null)
    nProcessONGPU=$(nvidia-smi | grep "nw.out" | wc -l)
    maxProcesses=1 # two process will be allowed to run simultaneously
    if [ "$nProcessONGPU" -le "$maxProcesses" ]; then
	if [[ -f $spkfile ]]; then
	    echo "      not launching"
	else
	    echo "----> LAUNCHING!!!!!"
            screen -dmS p${p}tau${1} ./nw.out 0 0 0 0
	    sleep 10
	    # break
	fi
	echo "============================================================================"
    else
	echo "reached max proceses on GPU"
        break
    fi
done
