#!/usr/bin/env python
#TO USE: rosrun teleop_keyboard teleop_keyboard.py
import roslib; roslib.load_manifest('blimp_control')
import rospy
from std_msgs.msg import Float64

import sys, select, termios, tty

msg = """
Reading from the keyboard  and Publishing to Float64!

---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

anything else : stop

i and , for pure linear
l and j for pure rotation
u o m . for combination
k for KILL all motors except altitude.
----------------------------
Altitude setting: (default is 1 meter)
t/b : increase/decrease Altitude #to be determined

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

e = """Error"""

moveBindings = {
        'i':(0.3,0.0),
        'o':(0.3,-0.3),
        'j':(0.0,0.3),
        'l':(0.0,-0.3),
        'k':(0.0,0.0),
        'u':(0.3,0.3),
        ',':(-0.3,0.0),
        '.':(-0.3,0.3),
        'm':(-0.3,-0.3),
        }

altitudeBindings={
        't':(0.01),
        'b':(-.01),
        }

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
        }

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(linear, angular, altitude):
    return "currently:\tlinear %s\tangular %s\taltitude %s " % (linear,angular, altitude)

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    pub0 = rospy.Publisher('cmd_alt', Float64, queue_size = 10)
    pub1 = rospy.Publisher('cmd_v', Float64, queue_size = 10)
    pub2 = rospy.Publisher('cmd_w', Float64, queue_size = 10)
    rospy.init_node('teleop_keyboard')
    rate = rospy.Rate(10)

    v = 0.0
    w = 0.0
    z = 1.0
    ik = -1.0

    status = 0

    try:
        print msg
        while not rospy.is_shutdown():
            key = getKey()
            if key in moveBindings.keys():
                v = moveBindings[key][0]
                w = moveBindings[key][1]
            elif key in speedBindings.keys():
                pass
            elif key in altitudeBindings.keys():
                z = z + moveBindings[key][0]
            else:
                v = 0.0
                w = 0.0
                # does it need a case for z?
                if (key == '\x03'):
                    break

            print 'v %f '%(v)
            print 'w %f '%(w)
            pub0.publish(z)
            pub1.publish(v)
            pub2.publish(w)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

    finally:
        print ik
        pub0.publish(z)
        pub1.publish(v)
        pub2.publish(w)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

