#!/bin/bash

host_ip=10.0.1.37
port=2222

echo Starting raspivid camera stream
echo Streaming to $host_ip:$port

# Process is backgrounded with the & operator in Bash
raspivid -t 0 -w 640 -h 480 -hf -fps 40 -o tcp://$host_ip:$port

# echo Running raspivid camera stream at PID $!

