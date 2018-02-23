#!/usr/bin/env python
#tflot


import rospy
from std_msgs.msg import Float64
import propeller
import time

# Initialize and configure blimp
print("Connecting to the propeller")
prop = propeller.Prop()

u_scale = 32767
u0 = 0.0
u = 0.0

def down_callback(data):
    global u
    u = data.data
    if u > 1:
        u = 1
    prop.down(int(1.2*u*u_scale))
    #print(data)

def v_callback(data):
    global u0
    u0 = data.data
    
def delta_callback(data):
    global u0
    u_delta = data.data
    #prop.right(int(-1*(-u0 - u_delta-u*0.5)*u_scale))#its the right prop in the test setup
    #prop.left(int(-1*(-u0+ u_delta+u*0.5)*u_scale))# its the left set up in the test set u
    
    prop.right(int(-1*(-u0 - u_delta)*u_scale))#its the right prop in the test setup
    prop.left(int(-1*(-u0+ u_delta)*u_scale))# its the left set up in the test set u

def propellers():
    rospy.init_node("propellers", anonymous=True)
    rospy.Subscriber("prop_down_re", Float64, down_callback)
    rospy.Subscriber("v_setpoint", Float64, v_callback)
    rospy.Subscriber("delta_re", Float64, delta_callback)
    rospy.spin()

if __name__ == '__main__':
    propellers()
