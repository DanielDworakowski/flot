#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Float64

def talker():
    pub0 = rospy.Publisher('alt_setpoint', Float64, queue_size=10)
    pub1 = rospy.Publisher('v_setpoint', Float64, queue_size=10)
    pub2 = rospy.Publisher('w_setpoint', Float64, queue_size=10)
    rospy.init_node('dummy_setpoint', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub0.publish(1.0)
        pub1.publish(0.0)
        pub2.publish(0.0)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
