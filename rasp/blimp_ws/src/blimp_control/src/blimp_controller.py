#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped
from PID import Controller

def main():
    rospy.init_node('blimp_controller', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    controller = Controller()
    while not rospy.is_shutdown():
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass