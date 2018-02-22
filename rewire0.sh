#!/bin/bash
set -x
kappa=$1
rewiredEEWeight=$2
trNo=$3

XI=0.3

# mExt=$1
# mExtOne=$2
p=0
gamma=0


JIFACTOR=1
N=10000
K=1000
T_STOP=21000
TSTOP0=11000

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

tStringTmp0="$(echo "$TSTOP0*$mille" | bc)";
tString0=${tStringTmp0%.*}


baseFldr='/homecentral/srao/cuda'
trNo0=0
python WritePOToFile.py 0 ${trNo0} ${K} ${TSTOP0} ${XI}

# for trNo in 10;
# do
    echo $kappa
    cd $baseFldr
    make clean
    IF_REWIRE=1
    eval sed "-i 's^#define NE .*^#define NE ${N}ULL^' devHostConstants.h"
    eval sed "-i 's^#define NI .*^#define NI ${N}ULL^' devHostConstants.h"
    eval sed "-i 's^#define K .*^#define K ${K}.0^' devHostConstants.h"
    eval sed "-i 's^#define TSTOP .*^#define TSTOP ${T_STOP}^' devHostConstants.h"
    eval sed "-i 's^#define IF_REWIRE .*^#define IF_REWIRE ${IF_REWIRE}^' devHostConstants.h"
    # eval sed "-i 's^#define rewiredEEWeight .*^#define rewiredEEWeight ${rewiredEEWeight}^' devHostConstants.h"
    # eval sed "-i 's^#define kappa .*^#define kappa ${kappa}^' devHostConstants.h"    

    kStringTmp="$(echo "${kappa}*$ten" | bc)";
    kString=${kStringTmp%.*}
    fldr="${baseFldr}/data/rewire/N${N}K${K}/kappa${kString}/p${pString}gamma${gString}/T${tString}/tr${trNo}"
    mkdir -p $fldr
    fldrForTr0="${baseFldr}/data/rewire/N${N}K${K}/kappa0/p${pString}gamma${gString}/T${tString0}/tr${trNo0}/*.dat"
    echo $fldrForTr0
    make    
    mv a.out $fldr
    mv nw.out $fldr
    for THETA in {0..90..5};
    do	
	cd $fldr
	pwd
	if [ $THETA == 0 ] 
	then
            ln -snf -t $fldr $fldrForTr0
	    ./a.out 0 ${kappa} ${rewiredEEWeight}
	    screen -dm ./nw.out 0 0 ${THETA} ${trNo} ${rewiredEEWeight} 
	    sleep 28m
	else
	    screen -dm ./nw.out 0 0 ${THETA} ${trNo} ${rewiredEEWeight}
	    sleep 28m
	fi
    done
    cd $baseFldr
    sleep 5m    
# done
