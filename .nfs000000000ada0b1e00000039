#!/bin/bash
set -x

trNo=$1

echo $trNo



K=1000
p=0
gamma=0
N=10000
JIFACTOR=1.0
T_STOP=10000
thousand=1000
# m0StringTmp="$(echo "$mExt*$thousand" | bc)";
# m0String=${m0StringTmp%.*}

# m1StringTmp="$(echo "$mExtOne*$thousand" | bc)";
# m1String=${m1StringTmp%.*}


ten=10
pStringTmp="$(echo "$p*$ten" | bc)";
pString=${pStringTmp%.*}
gStringTmp="$(echo "${gamma}*$ten" | bc)";
gString=${gStringTmp%.*}

mille=0.001
tStringTmp="$(echo "$T_STOP*$mille" | bc)";
tString=${tStringTmp%.*}



baseFldr='/homecentral/srao/cuda'
nTrials=10
IF_REWIRE=0
IF_SAVE=0


# for trNo in 1999; #{100..111};
# do
    cd $baseFldr
    make clean
    eval sed "-i 's^#define NE .*^#define NE ${N}ULL^' devHostConstants.h"
    eval sed "-i 's^#define NI .*^#define NI ${N}ULL^' devHostConstants.h"
    # eval sed "-i 's^#define NFF .*^#define NFF ${N}^' devHostConstants.h"
    eval sed "-i 's^#define K .*^#define K ${K}.0^' devHostConstants.h"
    eval sed "-i 's^#define TSTOP .*^#define TSTOP ${T_STOP}^' devHostConstants.h"
    eval sed "-i 's^#define IF_REWIRE .*^#define IF_REWIRE ${IF_REWIRE}^' devHostConstants.h"
    make
    fldr="${baseFldr}/data/rewire/N${N}K${K}/kappa0/p${pString}gamma${gString}/T${tString}/tr${trNo}"
    mkdir -p $fldr
    mv genconvec.out $fldr
    mv nw.out $fldr    
    cd $fldr
    pwd
    echo ${IF_SAVE}    
    for THETA in 0 22.5 45 67.5 90 112.5 135 157.5 180;
    do
	echo ${IF_SAVE}
	if [ $THETA == 0 ] 
	then
	    ./genconvec.out 0
	    screen -dm ./nw.out 0 0 ${THETA} ${trNo}
	    # screen -dmS k0trNo${trNo}phi${THETA} ./nw.out ${DEVICE_ID} 0 ${THETA} ${trNo}
	    sleep 14m
	else
	    screen -dm ./nw.out 0 0 ${THETA} ${trNo}
	    # screen -dmS k0trNo${trNo}phi${THETA} ./nw.out ${DEVICE_ID} 0 ${THETA} ${trNo}
	    sleep 14m
	fi
    done
    # sleep 10m
# done
