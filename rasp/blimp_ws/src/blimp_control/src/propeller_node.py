#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
import propeller
import time

# Initialize and configure blimp
print("Connecting to the propeller")
prop = propeller.Prop()

u_scale = 32767
u0 = 0.0

def down_callback(data):
    u = data.data
    if u > 1:
        u = 1
    prop.down(int(u*u_scale))
    #print(data)

def v_callback(data):
    global u0
    u0 = data.data

def delta_callback(data):
    global u0
    u_delta = data.data
    prop.left(int(u0*u_scale + u_delta))
    prop.right(int(u0*u_scale - u_delta))

def propellers():
    rospy.init_node("propellers", anonymous=True)
    rospy.Subscriber("prop_down_re", Float64, down_callback)
    rospy.Subscriber("v_setpoint", Float64, v_callback)
    rospy.Subscriber("delta_re", Float64, delta_callback)
    rospy.spin()

if __name__ == '__main__':
    propellers()
