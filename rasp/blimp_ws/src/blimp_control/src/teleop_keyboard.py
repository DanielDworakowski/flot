#!/usr/bin/env python
#TO USE: rosrun teleop_keyboard teleop_keyboard.py
from roslib import load_manifest; load_manifest('blimp_control')
from rospy import Publisher, init_node, Rate, is_shutdown
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

fixedV = 0.3
fixedW = 0.12

moveBindings = {
        'i':(fixedV,0.0),
        'o':(0.1,-1*fixedW),
        'j':(0.0,fixedW),
        'l':(0.0,-1*fixedW),
        'k':(0.0,0.0),
        'u':(0.1,fixedW),
        ',':(-1*fixedV,0.0),
        '.':(-0.1,fixedW),
        'm':(-0.1,-1*fixedW),
        }

altitudeBindings={
        't':(0.01),
        'b':(-0.01),
        }

vSpeedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        }

wSpeedBindings={
	'e':(1.1,1.1),
	'c':(.9,.9),
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

    pub_a = Publisher('cmd_alt', Float64, queue_size = 10)
    pub_v = Publisher('cmd_v', Float64, queue_size = 10)
    pub_w = Publisher('cmd_w', Float64, queue_size = 10)
    init_node('teleop_keyboard')
    rate = Rate(10)

    v = 0.0
    w = 0.0
    z = 1.0

    try:
        print msg
        while not is_shutdown():
            key = getKey()
            if key in moveBindings.keys():
                v = moveBindings[key][0]
                w = moveBindings[key][1]
            elif key in vSpeedBindings.keys():
		fixedV = fixedV*vSpeedBindings[key][0]
	    elif key in wSpeedBindings.keys():
		fixedW = fixedW*wSpeedBindings[key][0]                   
            elif key in altitudeBindings.keys():
                z += altitudeBindings[key]
                z = max(min(1.75,z),0.2)
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

    except ROSInterruptException:
        pass

    finally:
        print 'IM SHOOKETH'
        pub_a.publish(z)
        pub_v.publish(v)
        pub_w.publish(w)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

