#!/usr/bin/env python

from rospy import init_node, Subscriber, spin
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
    init_node('propellers', anonymous=True)
    Subscriber("prop_down", Float64, down_callback)
    spin()


if __name__ == '__main__':
    propellers()