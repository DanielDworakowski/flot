#!/usr/bin/env bash

foldername=$(date +%Y%m%d-%H%M%S)
echo $foldername
mkdir $foldername
cd $foldername

airsim_log=airsim_$foldername.log
python_log=python_agent_$foldername.log

COUNT=1

while :
do
    echo "run $COUNT"
    echo -e "\nrun $COUNT\n" >> $airsim_log
    echo -e "\nrun $COUNT\n" >> $python_log
    ########################################## CHANGE THIS DIRECTORY. NOTE: that it isn't the OrangeRoom.sh, you have to go 2 more folders down to just "OrangeRoom"
    /home/ddworakowski/flot/workspace/testEnvs/orange/OrangeRoom/Binaries/Linux/OrangeRoom OrangeRoom -windowed >> $airsim_log 2>&1 &
    AIRSIM_PID=$!
    sleep 5s

    ########################################## CHANGE THIS DIRECTORY
    timeout -sHUP 10m python ~/flot/workspace/runAgent.py --agent=dumbAgent --config=DefaultConfig >> $python_log 2>&1
    #python ~/flot/workspace/runAgent.py --agent=dumbAgent --config=DefaultConfig >> $python_log 2>&1
    ps aux | grep -i runAgent

    echo -e "CRASHED\n\n"
    kill -9 $AIRSIM_PID
    sleep 7s

    COUNT=$(( $COUNT + 1  ))

done
