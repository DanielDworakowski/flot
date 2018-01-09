#!/bin/bash

host_ip=10.0.1.26
port=2224

echo Starting raspivid camera stream
echo Streaming to $host_ip:$port

# Process is backgrounded with the & operator in Bash
raspivid -t 0 -w 640 -h 480 -hf -fps 40 -o tcp://10.0.1.26:2224 &

echo Running raspivid camera stream at PID $!

