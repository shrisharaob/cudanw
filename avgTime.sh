#!/bin/sh
set -eu # makes your program exit on error or unbound variable
GPU=0
IF_SAVE=0
make clean
make
for n in {1..100}
do
    echo $n
    ./a.out $GPU $IF_SAVE
done


