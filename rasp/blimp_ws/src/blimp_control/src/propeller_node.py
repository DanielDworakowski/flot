#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
import Blimp

# Initialize and configure blimp
blimp = Blimp.Blimp()
print("Connecting to the propellers")
blimp.connect()
print("Propellers Connected!")

u_scale = 32768

def down_callback(data):
	blimp.down(data.data*u_scale)

def propellers():
    rospy.init_node('propellers', anonymous=True)
    rospy.Subscriber("prop_down", Float64, down_callback)
    rospy.spin()

if __name__ == '__main__':
	propellers()
