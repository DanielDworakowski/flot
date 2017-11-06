__author__ = 'Kelvin Chan'
__version__ = '0.1'

import sys, io

# Image libraries
import numpy as np
from PIL import Image

# ROS Libraries
import rospy

# ROS Messages
from sensor_msgs.msg import CompressedImage

def image_callback(ros_data):
    
    print("Image received")
    # print(ros_data)
    # arr = np.fromstring(ros_data.data, np.uint8)
    # print(arr)
    # img = Image.fromarray(arr)
    # img.show()

image_sub = rospy.Subscriber("/output/image_raw/compressed",
                            CompressedImage, image_callback,
                            queue_size = 1)

rospy.init_node('image_subscriber', anonymous=True)

print("Started image_subscriber node...")

try:
   rospy.spin()
except KeyboardInterrupt:
   print("Shutting down image_subscriber...")
