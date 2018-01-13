#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
import propeller
import time

# Initialize and configure blimp
print("Connecting to the propeller")
prop = propeller.Prop()

u_scale = 32767

def down_callback(data):
    u = data.data
    if u > 1:
        u = 1
    prop.down(int(u*u_scale))
    #print(data)

def propellers():
    rospy.init_node("propellers", anonymous=True)
    rospy.Subscriber("prop_down_re", Float64, down_callback)
    rospy.spin()

if __name__ == '__main__':
    propellers()
