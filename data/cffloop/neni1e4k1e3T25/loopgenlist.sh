#!/bin/bash

for i in `seq 2 9`;
do
  ./genlist.sh 1 ${i}.0
  mv list.txt l${i}.txt
  mv l${i}.txt ~/db
done
