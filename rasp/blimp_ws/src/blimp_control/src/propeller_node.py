#!/usr/bin/env python
#tflot

from rospy import init_node, Subscriber, spin
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
    global u0
    u = data.data
    if u > 1:
        u = 1
    prop.down(int((u+0.9*u0)*u_scale))
    #print(data)

def v_callback(data):
    global u0
    u0 = data.data
    
def delta_callback(data):
    global u0
    global u
    u_delta = data.data
    prop.right(int(-1*(-u0 - u_delta-u*0.55)*u_scale))#its the right prop in the test setup
    prop.left(int(-1*(-u0+ u_delta+u*0.55)*u_scale))# its the left set up in the test set u

def propellers():
    #rospy.init_node("propellers", anonymous=True)
    #rospy.Subscriber("prop_down_re", Float64, down_callback)
    #rospy.Subscriber("cmd_v", Float64, v_callback)
    #rospy.Subscriber("delta_re", Float64, delta_callback)
    #rospy.spin()

    init_node("propellers", anonymous=True)
    Subscriber("prop_down", Float64, down_callback)
    Subscriber("cmd_v", Float64, v_callback)
    #Subscriber("blimp_vt", Float64, v_callback)
    Subscriber("delta", Float64, delta_callback)
    spin()

if __name__ == '__main__':
    propellers()
