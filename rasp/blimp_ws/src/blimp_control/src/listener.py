#!/usr/bin/env python

from __future__ import print_function

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

import sys
import os
# import rospy
from rospy import init_node, Subscriber, Rate, spin
import datetime
from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from subprocess import Popen
from blimp_control.msg import Float64WithHeader
from tf.transformations import euler_from_quaternion

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
        # eprint('sonar:{}\n'.format(data.header))
        val = data.float.data
        with open(sfile,'a') as f:
            f.write('{},{}\n'.format(val, stamp))

    elif datatype=='imu':
        # eprint('imu:{}\n'.format(data))
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
        # eprint('yaw:{}\n'.format(data.header))
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
    eprint('Building Listener.')
    init_node('listener', anonymous=True)
    Subscriber('sonar_meas', Float64WithHeader, sonar_callback)
    Subscriber('imu_data', Imu, imu_callback)
    Subscriber('yaw_rate', Float64WithHeader, yaw_callback)
    rate = Rate(100)   # 100 Hz to prevent aliasing of 40 FPS feed

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
    eprint('Opening camera logging.')
    pipe = Popen(raspividcmd)
    eprint('Spinning.')
    # spin() simply keeps python from exiting until this node is stopped
    spin()

if __name__ == '__main__':
    listener()
