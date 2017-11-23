#!/usr/bin/env python
# __author__ = 'Kelvin Chan'
# __version__ = '0.1'

import sys, io

# RPi Camera Library
import picamera

# Image libraries
import numpy as np
from PIL import Image

# ROS Libraries
import rospy

# ROS Messages
from sensor_msgs.msg import CompressedImage

# Init camera and output RGB array
camera = picamera.PiCamera()

def main():
    image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size = 1)
    rospy.init_node('ROSPiCam')
    rate = rospy.Rate(10) # 10hz
    # Set publisher node to 15 Hz
    msg = CompressedImage()
    msg.format = "jpeg"
    # Publisher node loop
    while not rospy.is_shutdown():
        # Open new BytesIO stream
        stream = io.BytesIO()

        # Capture frame from camera
        camera.capture(stream, format='jpeg')
        # camera.capture("/home/pi/temp.jpg")

        msg.header.stamp = rospy.Time.now()
        msg.data = np.array(Image.open(stream)).tostring()

        # Publish image
        image_pub.publish()

        # Close stream
        stream.close()
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

