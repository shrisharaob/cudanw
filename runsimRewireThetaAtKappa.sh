#!/bin/bash
#set -x
# mExt=$1
# mExtOne=$2
p=0
gamma=0
kappa=$1
rewiredWeight=$2

JIFACTOR=1
N=10000
K=1000
T_STOP=10000


thousand=1000

m0StringTmp="$(echo "$mExt*$thousand" | bc)";
m0String=${m0StringTmp%.*}

m1StringTmp="$(echo "$mExtOne*$thousand" | bc)";
m1String=${m1StringTmp%.*}

ten=10
pStringTmp="$(echo "$p*$ten" | bc)";
pString=${pStringTmp%.*}
gStringTmp="$(echo "${gamma}*$ten" | bc)";
gString=${gStringTmp%.*}

mille=0.001
tStringTmp="$(echo "$T_STOP*$mille" | bc)";
tString=${tStringTmp%.*}

baseFldr='/homecentral/srao/cuda'
trNo0=0
python WritePOToFile.py 0 ${mExt} ${mExtOne} ${trNo0} ${K}

for trNo in 10;
do
    echo $kappa
    cd $baseFldr
    make clean
    IF_REWIRE=1
    eval sed "-i 's^#define NE .*^#define NE ${N}ULL^' devHostConstants.h"
    eval sed "-i 's^#define NI .*^#define NI ${N}ULL^' devHostConstants.h"
    eval sed "-i 's^#define K .*^#define K ${K}.0^' devHostConstants.h"
    eval sed "-i 's^#define TSTOP .*^#define TSTOP ${T_STOP}^' devHostConstants.h"
    eval sed "-i 's^#define IF_REWIRE .*^#define IF_REWIRE ${IF_REWIRE}^' devHostConstants.h"
    eval sed "-i 's^#define rewiredEEWeight .*^#define rewiredEEWeight ${rewiredEEWeight}^' devHostConstants.h"    
    make
    kStringTmp="$(echo "${kappa}*$ten" | bc)";
    kString=${kStringTmp%.*}
    fldr="${baseFldr}/data/rewire/N${N}K${K}/kappa${kString}/p${pString}gamma${gString}/T${tString}/tr${trNo}"
    mkdir -p $fldr
    fldrForTr0="${baseFldr}/data/rewire/N${N}K${K}/kappa0/p${pString}gamma${gString}/T${tString}/tr${trNo0}/*.dat"	    
    mv a.out $fldr
    mv 
    for phi in 0 22.5 45 67.5 90 112.5 135 157.5 180;
    do	
	cd $fldr
	pwd
        ln -snf -t $fldr $fldrForTr0
	screen -dm ./nw.out 0 0 ${THETA} ${trNo}
    done
    sleep 15m
    cd $baseFldr    
done
