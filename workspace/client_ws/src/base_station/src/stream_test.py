#!/usr/bin/env python

from subprocess import Popen, PIPE
from shlex import split

# ROS Libraries
import rospy
import roslib

VERBOSE=False

# Shell commands
# Listen at port for TCP stream and pipe to next
command1 = "nc -l 2222"
# Play back stream from stdin
command2 = "mplayer -fps 20 -demuxer h264es -"

def stream_test():

    p1 = Popen(split(command1), stdout=PIPE)
    p2 = Popen(split(command2), stdin=p1.stdout)

    while True:
        pass

if __name__ == '__main__':
    try:
        stream_test()

    except rospy.ROSInterruptException:
        if VERBOSE:
            print('Node was interrupted; shutting down node...')
        pass

    finally:
        if VERBOSE:
            print('Closing windows...')
