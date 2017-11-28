#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

from cv_bridge import CvBridge

image_pub = rospy.Publisher("/pi_image", Image, queue_size=1)

def callback(ros_data):
    np_arr = np.fromstring(ros_data.data, np.uint8).reshape([1080,1920,3])
    image_pub.publish(bridge.cv2_to_imgmsg(np_arr, "bgr8"))

image_sub = rospy.Subscriber("output/image_raw/compressed", CompressedImage, callback)
bridge = CvBridge()

def main():
    rospy.init_node('ROSPiCamSlaveRepublish')
    rate = rospy.Rate(100) # 10hz
    while not rospy.is_shutdown():
        rate.sleep()
        rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
