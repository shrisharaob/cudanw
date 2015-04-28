for i in 0 1 2 3 4 5 6 7 8 9 10; 
do

#    th=(0 22 45 67 90 112 135 157);
#    th=0;
    CONTRAST=(0 10 20 30 40 50 60 70 80 90 100);
    th=(0 10 20 30 40 50 60 70 80 90 100);
    thTrue=(0. 22.5 45. 67.5 90. 112.5 135. 157.5);
    SIMDURATION=30000 #IN MS
    NTRIALS=$1
    echo "real theta: " ${th[i]}
    for ((ii=0; $ii<NTRIALS; ii=$ii+1))
    do
	echo spkTimes_xi1.2_theta${th[0]}_0.00_3.0_cntrst${CONTRAST[i]}_${SIMDURATION}_tr${ii}.csv
        for fl in `ls spkTimes_xi1.2_theta${th[0]}_0.00_3.0_cntrst${CONTRAST[i]}_${SIMDURATION}_tr${ii}.csv`; 
        do
            jj=$(($ii+1))
            kk=$(($jj * 1000))
#            echo "trial: " $jj
            kk=$(($kk+${th[$i]}))
 #           echo "db theta: " $kk
            echo $fl $kk >> list.txt
        done
    done
done
