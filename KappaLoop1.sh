#!/bin/bash
set -x

ten=10
rewiredEEWeight=5
trNo="$(echo "${rewiredEEWeight}*$ten" | bc)";

echo ${trNo}

for kappa in 2 4;
do
    screen -dm ./runsimRewireThetaAtKappa.sh ${kappa} ${rewiredEEWeight} ${trNo}
    sleep 125m
done
