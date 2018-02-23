#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import os
import rospy
import datetime
from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu

import sys, time, threading
import numpy as np
from subprocess import Popen, PIPE
from shlex import split
from blimp_control.msg import Float64WithHeader
from tf.transformations import euler_from_quaternion

# Removes conflict regarding CV2 if ROS Kinetic has been sourced
del os.environ['PYTHONPATH']

timestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(timestr):
    os.makedirs(timestr)

sfile = '{}/sonar-observation.csv'.format(timestr)
ifile = '{}/imu-observation.csv'.format(timestr)
yfile = '{}/yawrate-observation.csv'.format(timestr)

if os.path.isfile(sfile): os.remove(sfile)
if os.path.isfile(ifile): os.remove(ifile)
if os.path.isfile(yfile): os.remove(yfile)

fs = open(sfile, 'a')
fi = open(ifile, 'a')
fy = open(yfile, 'a')

with open(sfile,'a') as f:
    f.write('Value, Timestamp\n')
with open(ifile,'a') as f:
    f.write('AngVel(x), AngVel(y), AngVel(z), LinAcc(x), LinAcc(y), LinAcc(z), Roll, Pitch, Yaw, Timestamp\n')
with open(yfile,'a') as f:
    f.write('Value, Timestamp\n')

def store(data, datatype):
    ts = data.header.stamp.secs
    tn = data.header.stamp.nsecs/float(1e9)
    stamp = "{:.10f}".format(ts+tn)
    if datatype=='sonar':
        print('sonar:{}\n'.format(data.header))
        val = data.float.data
        with open(sfile,'a') as f:
            f.write('{},{}\n'.format(val, stamp))

    elif datatype=='imu':
        print('imu:{}\n'.format(data))
        quat = data.orientation
        angvel = data.angular_velocity
        linacc = data.linear_acceleration
        roll, pitch, yaw = euler_from_quaternion([quat.w,quat.x,quat.y,quat.z])
        st = '{},{},{},{},{},{},{},{},{}'.format(
                angvel.x,
                angvel.y,
                angvel.z,
                linacc.x,
                linacc.y,
                linacc.z,
                roll,
                pitch,
                yaw
                )
        with open(ifile,'a') as f:
            f.write('{},{}\n'.format(st,stamp))

    elif datatype=='yaw':
        print('yaw:{}\n'.format(data.header))
        val = data.float.data
        with open(yfile,'a') as f:
            f.write('{},{}\n'.format(val, stamp))

    cwd = os.getcwd()

def sonar_callback(data):
    store(data, 'sonar')

def imu_callback(data):
    store(data, 'imu')

def yaw_callback(data):
    store(data, 'yaw')

def cam_callback(data):
    store(data, 'cam')

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    print('Building Listener.')
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('sonar_meas', Float64WithHeader, sonar_callback)
    rospy.Subscriber('imu_data', Imu, imu_callback)
    rospy.Subscriber('yaw_rate', Float64WithHeader, yaw_callback)
    rate = rospy.Rate(100)   # 100 Hz to prevent aliasing of 40 FPS feed

    raspividcmd = [
        'raspivid',
        '-t',
        '0',
        '-w',
        '640',
        '-h',
        '480',
        '-hf',
        '-vf',
        '-fps',
        '30',
        '-o',
        '%s/video.h264'%(timestr),
        '--save-pts',
        '%s/video_ts.csv'%(timestr)
    ]
    print('Opening camera logging.')
    pipe = Popen(raspividcmd)
    print('Spinning.')
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
