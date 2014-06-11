#!/bin/sh
set -eu # makes your program exit on error or unbound variable
for n in {100..100..500}
do
    eval sed "-i 's_#define K .*_#define K $n.0_' devHostConstants.h"
    make clean
    make
    ./a.out
done



