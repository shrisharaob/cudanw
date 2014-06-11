#!/bin/bash
set -eu # makes your program exit on error or unbound variable
set -x # verbose
for n in {20..50..20}
do
    eval sed "-i 's_#define NI .*_#define NI ${n}_' devHostConstants.h"
  #  make clean
 #   make 
#    ./a.out
done



