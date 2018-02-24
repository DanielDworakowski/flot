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
        'o':(0.12,-0.12),
        'j':(0.0,0.12),
        'l':(0.0,-0.12),
        'k':(0.0,0.0),
        'u':(0.12,0.12),
        ',':(-0.3,0.0),
        '.':(-0.12,0.12),
        'm':(-0.12,-0.12),
        }

altitudeBindings={
        't':(0.01,0),
        'b':(-0.01,0),
        }

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.2,1),
        'x':(.8,1),
        'e':(1.3,1.1),
        'c':(0.7,.9),
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

    pub_a = rospy.Publisher('cmd_alt', Float64, queue_size = 10)
    pub_v = rospy.Publisher('cmd_v', Float64, queue_size = 10)
    pub_w = rospy.Publisher('cmd_w', Float64, queue_size = 10)
    rospy.init_node('teleop_keyboard')
    rate = rospy.Rate(10)

    v = 0.0
    w = 0.0
    z = 1.0

    try:
        print msg
        while not rospy.is_shutdown():
            key = getKey()
            if key in moveBindings.keys():
                v =moveBindings[key][0]
                w = moveBindings[key][1]
            elif key in speedBindings.keys():
                pass
            elif key in altitudeBindings.keys():
                z += altitudeBindings[key][0]
                z = max(min(1.75,z),0)
                #z = z + moveBindings[key][0]
                #print 'HER'
            else:
                v = 0.0
                w = 0.0
                if (key == '\x03'):
                    break

            print 'v %f w %f z %f m \n'%(v,w,z)
            pub_a.publish(z)
            pub_v.publish(v)
            pub_w.publish(w)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

    finally:
        print 'IM SHOOKETH'
        pub_a.publish(z)
        pub_v.publish(v)
        pub_w.publish(w)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

