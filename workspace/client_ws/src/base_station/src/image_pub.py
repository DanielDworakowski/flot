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

# Shell commands
command1 = split("nc -l 2222")
command2 = [ 'ffmpeg',
        '-i', 'pipe:0',             # fifo is the named pipe
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
msg.format = 'png'

def image_pub(v=True):

    # Listen for TCP stream and feed into FFMPEG for converting into image
    if v:
        print('Listening for TCP video stream and converting to images...')

    nc_pipe = Popen(command1, stdout=PIPE)
    pipe = Popen(command2, stdin=nc_pipe.stdout, stdout=PIPE, bufsize=bufsize)

    # ROS Node setup
    if v:
        print('Starting image_pub node...')

    client = RobotUtil.VideoStreamClient(VERBOSE=VERBOSE, BGR2RGB=True)
    client.start()

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

        if v:
            if image is not None:
                cv2.imshow('Video', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Flush pipe for new messages
        pipe.stdout.flush()

        # Publish compressed image with new timestamp
        image = client.frame

        if image is not None:
            msg.header.stamp = rospy.Time.now()
            msg.data = np.array(cv2.imencode('.png', image)[1]).tostring()
            pub.publish(msg)

        rate.sleep()    # Maintain loop rate

if __name__ == '__main__':
    VERBOSE = True
    try:
        image_pub(VERBOSE)

    except rospy.ROSInterruptException:
        if VERBOSE:
            print('Node was interrupted; shutting down node...')
        pass

    finally:
        if VERBOSE:
            print('Closing windows...')

        cv2.destroyAllWindows()
