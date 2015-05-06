#CONTRAST_LIST=(0 10 20 30 40 50 60 70 80 90 100);
CONTRAST_LIST=(100.0)
for kCntrst in 0; # 6 7 8 9 10;
do
    CONTRAST=${CONTRAST_LIST[kCntrst]}
    echo "contrast: " $CONTRAST
    for i in 0 1 2 3 4 5 6 7; 
    do
	th=(0 22 45 67 90 112 135 157);
	thTrue=(0. 22.5 45. 67.5 90. 112.5 135. 157.5);
	SIMDURATION=25000 #IN MS
	NTRIALS=$1
	echo "real theta: " ${th[i]}
	for ((ii=0; $ii<NTRIALS; ii=$ii+1))
	do
#	    echo spkTimes_xi1.2_theta${th[i]}_0.00_3.0_cntrst${CONTRAST}_${SIMDURATION}_tr${ii}.csv
            for fl in `ls spkTimes_xi1.2_theta${th[i]}_0.00_3.0_cntrst${CONTRAST}_${SIMDURATION}_tr${ii}.csv`; 
            do
		jj=$(($ii+1))
		kk=$(($jj * 1000))
		kk=$(($kk+${th[$i]}))
		echo $fl $kk >> list_cntrst$CONTRAST.txt
            done
	done
    done
done