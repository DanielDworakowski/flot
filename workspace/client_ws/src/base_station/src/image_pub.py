#!/usr/bin/env python

"""ROS Kinetic CompressedImage Publisher in python

Receives TCP stream from Raspberry Pi Camera
Publishes image as CompressedImage under /PI_CAM/image_raw/compressed topic

Written by: Kelvin Chan
"""

import sys, time
import numpy as np
import cv2
import subprocess as sp

# ROS Libraries
import rospy
import roslib

from sensor_msgs.msg import CompressedImage

VERBOSE=False

# Shell commands
NETCAT_BIN = "nc"
FFMPEG_BIN = "ffmpeg"

command = [NETCAT_BIN,
        '-l', '2224', '|'         # Listen at port for TCP stream and pipe to next
        FFMPEG_BIN,
        '-i', 'pipe:0',           # use stdin pipe
        '-pix_fmt', 'bgr24',      # opencv requires bgr24 pixel format.
        '-vcodec', 'rawvideo',
        '-an','-sn',              # we want to disable audio processing (there is no audio)
        '-f', 'image2pipe', '-']

# Pipe buffer size calculation for image size
width = 640
height = 480
depth = 3
num = 2
bufsize = width*height*depth*num

# CompressedImage Message Setup
msg = CompressedImage()
msg.format = 'jpeg'

def image_pub():

    # Listen for TCP stream and feed into FFMPEG for converting into image
    if VERBOSE:
        print('Listening for TCP video stream and converting to images...')

    pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=bufsize)

    # ROS Node setup
    pub = rospy.Publisher('/PI_CAM/image_raw/compressed', CompressedImage)
    rospy.init_node('image_pub', anonymous=True)
    rate = rospy.Rate(100)   # 100 Hz to prevent aliasing of 40 FPS feed

    # ROS loop
    while not rospy.is_shutdown():
        # Capture frame bytes from pipe
        raw_image = pipe.stdout.read(width*height*depth)

        # Transform bytes to numpy array
        image =  np.fromstring(raw_image, dtype='uint8')
        image = image.reshape((height, width, depth))

        if image is not None:
            cv2.imshow('Video', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Flush pipe for new messages
        pipe.stdout.flush()

        # Publish compressed image with new timestamp
        msg.header.stamp = rospy.Time.now()
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
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
