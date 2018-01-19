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
    if u0>1:
        u0=1
    elif u0<-1:
        u0=-1

def delta_callback(data):
    global u0
    # u_delta = data.data
    u_delta = 0.0
    prop.left(int(-u0 + u_delta)*u_scale)
    prop.right(int(-u0 - u_delta)*u_scale)

def propellers():
    rospy.init_node("propellers", anonymous=True)
    rospy.Subscriber("prop_down_re", Float64, down_callback)
    rospy.Subscriber("v_setpoint", Float64, v_callback)
    rospy.Subscriber("delta_re", Float64, delta_callback)
    rospy.spin()

if __name__ == '__main__':
    propellers()
