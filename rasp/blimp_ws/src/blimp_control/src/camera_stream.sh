#!/bin/bash

host_ip=10.0.1.101
port=2222

echo Starting raspivid camera stream
echo Streaming to $host_ip:$port

# Process is backgrounded with the & operator in Bash
name=$(date +%Y%m%d%H%M%S)
# raspivid -t 0 -w 640 -h 480 -hf -vf -fps 30 -o tcp://$host_ip:$port --save-pts $name.csv
raspivid -t 0 -w 640 -h 480 -hf -vf -fps 30 -o $name.h264 --save-pts 

# echo Running raspivid camera stream at PID $!

