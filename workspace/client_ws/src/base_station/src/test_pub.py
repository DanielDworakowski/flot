#!/usr/bin/env python

"""ROS Kinetic CompressedImage Publisher in python

Publishes some test images to the topic /test_image

Written by: Kelvin Chan
"""

import sys, time
import numpy as np
import cv2
from PIL import Image
from subprocess import Popen, PIPE
from shlex import split

# ROS Libraries
import rospy
import roslib

from sensor_msgs.msg import CompressedImage

VERBOSE = True

# CompressedImage Message Setup
msg = CompressedImage()
msg.format = 'png'

file_path = '/home/kelvin/Pictures/tau_at_005.png'

def image_pub():

    # ROS Node setup
    if VERBOSE:
        print('Starting test_pub node...')

    pub = rospy.Publisher('/test_image/compressed', CompressedImage)
    rospy.init_node('test_pub', anonymous=True)
    rate = rospy.Rate(100)   # 100 Hz to prevent aliasing of 40 FPS feed

    cnt = 0

    image = np.asarray(Image.open(file_path))

    # ROS loop
    while not rospy.is_shutdown():
        # Publish compressed image with new timestamp
        if image is not None:
            cnt += 1
            print(cnt)
            msg.header.stamp = rospy.Time.now()
            msg.data = np.array(cv2.imencode('.png', image)[1]).tostring()
            pub.publish(msg)

        # rate.sleep()    # Maintain loop rate

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
