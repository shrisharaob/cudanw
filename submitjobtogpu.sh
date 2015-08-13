#! /bin/bash

IS_DONE=false

while [ "$IS_DONE" != true ]; do
    echo "....."
    echo $cntr
    ((cntr+=1))
    if [ $cntr -gt 3 ]; then
	echo "inside if"
        IS_DONE=true
    fi
    sleep 1s
done

