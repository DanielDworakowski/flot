#!/usr/bin/env python

"""ROS Kinetic Blimp Command Publisher in python

Sends Float64 message for v_t, v_z, w to RaspberryPi
Starts up Pyro4 nameserver and daemon to create communication class to NN stack

Written by: Kelvin Chan
"""

# ROS Libraries
import rospy
import roslib

# ROS Messages
from std_msgs.msg import Float64

# Utils
from RobotControl import RobotCommands

VERBOSE = True

# Float64 Message Setup
msg = Float64()

# Robot Command Setuip
rc = RobotCommands()

def blimp_cmd_pub():
    rc.startup()

    # ROS Node setup
    if VERBOSE:
        print('Starting blimp_cmd_pub node...')

    rospy.init_node('blimp_cmd_pub', anonymous=True)
    rate = rospy.Rate(10)   # 100 Hz loop rate

    vt_pub = rospy.Publisher('/blimp_vt', Float64)
    # vz_pub = rospy.Publisher('/blimp_vz', Float64)
    alt_pub = rospy.Publisher('/blimp_alt', Float64)
    w_pub = rospy.Publisher('/blimp_w', Float64)

    # ROS loop
    while not rospy.is_shutdown():
        vt = rc.getVT()
        # vz = rc.getVZ()
        vz = 1.
        w = rc.getW()

        if vt is None:
            vt = 0.3
        if w is None:
            w = 0.1

        msg.data = float(vt)
        vt_pub.publish(msg)

        msg.data = float(vz)
        alt_pub.publish(msg)

        msg.data = float(w)
        w_pub.publish(msg)

        rate.sleep()    # Maintain loop rate

if __name__ == '__main__':
    try:
        blimp_cmd_pub()

    except rospy.ROSInterruptException:
        if VERBOSE:
            print('Node was interrupted; shutting down node...')
        pass

    finally:
        if VERBOSE:
            print('Cleaning up...')
        rc.cleanup()
