#!/usr/bin/env python
#tflot

import rospy
from std_msgs.msg import Float64
import propeller
import time

# Initialize and configure blimp
print("Connecting to the propeller")

prop = propeller.Prop()
msg = Float64()

u_scale = 32767
u0 = 0.0
u = 0.0

def down_callback(data):
    global u
    global u0
    #global u_delta
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
    #global u_delta
    u_delta = data.data
    prop.right(int(-1*(-u0 - u_delta-u*0.55)*u_scale))#its the right prop in the test setup
    prop.left(int(-1*(-u0+ u_delta+u*0.55)*u_scale))# its the left set up in the test set u

def propellers():
    #rospy.init_node("propellers", anonymous=True)
    #rospy.Subscriber("prop_down_re", Float64, down_callback)
    #rospy.Subscriber("cmd_v", Float64, v_callback)
    #rospy.Subscriber("delta_re", Float64, delta_callback)
    #rospy.spin()

    rospy.init_node("propellers", anonymous=True)

    # Publish battery level
    battery_pub = rospy.Publisher('/prop_battery', Float64)

    # Subscribe to command values
    rospy.Subscriber("prop_down", Float64, down_callback)
    rospy.Subscriber("cmd_v", Float64, v_callback)
    #Subscriber("blimp_vt", Float64, v_callback)
    rospy.Subscriber("delta", Float64, delta_callback)

    # rospy.spin()

    # 1 Hz refresh for battery level check
    rate = rospy.Rate(1)

    # ROS loop
    while not rospy.is_shutdown():
        # Update and publish battery level
        msg.data = float(prop.batteryLevel())
        battery_pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    propellers()
