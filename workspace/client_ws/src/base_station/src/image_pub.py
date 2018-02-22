#!/usr/bin/env python

"""ROS Kinetic CompressedImage Publisher in python

Receives TCP stream from Raspberry Pi Camera
Publishes image as CompressedImage under /PI_CAM/image_raw/compressed topic
Images are published under .jpeg format
To view the published images, use the following command:
rosrun image_view image_view _image_transport:=compressed image:=/PI_CAM/image_raw/

Ensure that the machine executing this program has access to roscore, as either
the host or client to the network

Ensure that this is running before streaming camera feed from Raspberry Pi
Video stream, and that the correct IP and port is given for the stream

Written by: Kelvin Chan
"""

import sys, time
import numpy as np
import cv2
from subprocess import Popen, PIPE
from shlex import split

from util import RobotUtil

# ROS Libraries
import rospy
import roslib

from sensor_msgs.msg import CompressedImage

VERBOSE = True

# CompressedImage Message Setup
msg = CompressedImage()
msg.format = 'png'

def image_pub():

    # ROS Node setup
    if VERBOSE:
        print('Starting image_pub node...')

    client = RobotUtil.VideoStreamClient(VERBOSE=VERBOSE, BGR2RGB=True)
    client.start()

    pub = rospy.Publisher('/PI_CAM/image_raw/compressed', CompressedImage)
    rospy.init_node('image_pub', anonymous=True)
    rate = rospy.Rate(100)   # 100 Hz to prevent aliasing of 40 FPS feed

    # ROS loop
    while not rospy.is_shutdown():
        # Publish compressed image with new timestamp
        image = client.getFrame()

        if image is not None:
            msg.header.stamp = rospy.Time.now()
            msg.data = np.array(cv2.imencode('.png', image)[1]).tostring()
            pub.publish(msg)

        rate.sleep()    # Maintain loop rate

if __name__ == '__main__':
    try:
        image_pub()

    except rospy.ROSInterruptException:
        if VERBOSE:
            print('Node was interrupted; shutting down node...')
        pass

    finally:
        if VERBOSE:
            print('Closing windows...')

        cv2.destroyAllWindows()
