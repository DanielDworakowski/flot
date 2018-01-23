#!/usr/bin/env python
#TO USE: rosrun teleop_keyboard teleop_keyboard.py
import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy

from geometry_msgs.msg import Float64

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

----------------------------
Altitude setting: (default is 1 meter)

t/b : increase/decrease Altitude #to be determined

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

moveBindings = {
		'i':(1.0,0.0),
		'o':(.7,-.7),
		'j':(0.0,1.0),
		'l':(0.0,-1.0),
        'k':(0.0,0.0),
		'u':(.7,.7),
		',':(-1.0,0.0),
		'.':(-.7,.7),
		'm':(-.7,-.7),
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

def teleop_talker():
    pub0 = rospy.Publisher('cmd_alt', Float64, queue_size = 10)
    pub1 = rospy.Publisher('cmd_v', Float64, queue_size = 10)
    pub2 = rospy.Publisher('cmd_w', Float64, queue_size = 10)
	rospy.init_node('teleop_keyboard')


if __name__=="__main__":
    	settings = termios.tcgetattr(sys.stdin)

	teleop_talker()

	linear = rospy.get_param("~linear", 1.0)
	angular = rospy.get_param("~angular", 0.0)
    altitude = rospy.get_param("~altitude", 1.0)

	v = 0.0
	w = 0.0
	z = 1.0

	status = 0

	try:
		print msg
		print vels(linear,angular, altitude)
		while(1):
			key = getKey()
			if key in moveBindings.keys():
				v = moveBindings[key][0]
				w = moveBindings[key][1]
			elif key in speedBindings.keys():
				linear = linear * speedBindings[key][0]
				angular = angular * speedBindings[key][1]
            elif key in altitudeBindings.keys():
				z = z + moveBindings[key][0]

				print vels(linear,angular, altitude)
				if (status == 14):
					print msg
				status = (status + 1) % 15
			else:
				v = 0.0
				w = 0.0
                # does it need a case for z?
				if (key == '\x03'):
					break

			pub0.publish(z)
            pub1.publish(v)
            pub2.publish(w)

	except:
		print e

	finally:
		pub0.publish(z)
        pub1.publish(v)
        pub2.publish(w)

    		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)



"""
def vels(speed,turn):
	return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    	settings = termios.tcgetattr(sys.stdin)

	pub0 = rospy.Publisher('cmd_alt', Float64, queue_size = 10)
    pub1 = rospy.Publisher('cmd_v', Float64, queue_size = 10)
    pub2 = rospy.Publisher('cmd_w', Float64, queue_size = 10)
	rospy.init_node('teleop_keyboard')

	speed = rospy.get_param("~speed", 0.5)
	turn = rospy.get_param("~turn", 1.0)
	x = 0
	y = 0
	z = 0
	th = 0
	status = 0

	try:
		print msg
		print vels(speed,turn)
		while(1):
			key = getKey()
			if key in moveBindings.keys():
				x = moveBindings[key][0]
				y = moveBindings[key][1]
				z = moveBindings[key][2]
				th = moveBindings[key][3]
			elif key in speedBindings.keys():
				speed = speed * speedBindings[key][0]
				turn = turn * speedBindings[key][1]

				print vels(speed,turn)
				if (status == 14):
					print msg
				status = (status + 1) % 15
			else:
				x = 0
				y = 0
				z = 0
				th = 0
				if (key == '\x03'):
					break

			twist = Twist()
			twist.linear.x = x*speed; twist.linear.y = y*speed; twist.linear.z = z*speed;
			twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th*turn
			pub.publish(twist)

	except:
		print e

	finally:
		twist = Twist()
		twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
		twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
		pub.publish(twist)

    		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

"""
