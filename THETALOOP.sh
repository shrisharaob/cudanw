#!/bin/bash
set -x

mExt=$1
mExtOne=$2
p=$3
gamma=$4
trNo=$5

for phi in 0 22.5 45 67.5 90 112.5 135 157.5;
do
    if [ $phi -eq 0 ]
    then
	screen -dmS phi${phi}p${p}g${gamma} ./a.out ${mExt} ${mExtOne} ${p} ${gamma} ${phi} ${trNo}
	sleep 2m
    else
	screen -dmS phi${phi}p${p}g${gamma} ./a.out ${mExt} ${mExtOne} ${p} ${gamma} ${phi} ${trNo}
    fi
done

