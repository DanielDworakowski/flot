#!/usr/bin/env python

import sys, time
import numpy as np
from PIL import Image
import cv2

import rospy
import roslib
from sensor_msgs.msg import CompressedImage

VERBOSE = False

videostream = None

def sub_callback(ros_data):

    np_arr = np.fromstring(ros_data.data, np.uint8)
    np_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    if VERBOSE:
        print('Received image type: "{}"'.format(ros_data.format))

    cv2.imshow('Sub_Video', np_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

def image_sub_stream():
    # ROS Node setup
    if VERBOSE:
        print('Starting image_sub_stream node...')
        print('Using CV version: {}'.format(cv2.__version__))

    subscriber = rospy.Subscriber('/PI_CAM/image_raw/compressed',
                    CompressedImage, sub_callback, queue_size=1)

    rospy.init_node('image_sub_stream', anonymous=True)
    rate = rospy.Rate(100)   # 100 Hz to prevent aliasing of 40 FPS feed

    # Prevent program from exiting
    rospy.spin()

if __name__ == '__main__':
    try:
        image_sub_stream()

    except rospy.ROSInterruptException:
        if VERBOSE:
            print('Node was interrupted; shutting down node...')
        pass

    finally:
        if VERBOSE:
            print('Closing windows...')
            cv2.destroyAllWindows()
