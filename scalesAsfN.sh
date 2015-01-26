#!/bin/sh
set -eu # makes your program exit on error or unbound variable
for n in {800..10000..200}
do
    eval sed "-i 's_#define NI .*_#define NI ${n}_' devHostConstants.h"
    cat devHostConstants.h
    #make clean
    #make
    #./a.out
done



