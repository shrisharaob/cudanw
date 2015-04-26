#!/bin/bash
set -eu
set -x
for n in $(< $1)
do
    echo $n
    eval sed "-i 's^#define HOST_CONTRAST .*^#define HOST_CONTRAST ${n}^' devHostConstants.h"    
    nvcc -arch=sm_35 -O9 mysolver.cu -o nw_alpha$n.out
    ./nw_alpha$n.out
done
