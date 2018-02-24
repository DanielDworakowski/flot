#!/usr/bin/env python

# Modified from:
# https://www.raspberrypi.org/forums/viewtopic.php?t=84494

import time
import pigpio
from rospy import Publisher, init_node, Rate, is_shutdown, get_rostime, ROSInterruptException
from std_msgs.msg import Float64
import RPi.GPIO as GPIO
import time
from blimp_control.msg import Float64WithHeader

TRIGGER=23
ECHO=24
pub = Publisher('sonar_meas', Float64WithHeader, queue_size=10)
pub1 = Publisher('sonar_meas_control', Float64, queue_size=10)
high_tick = None # global to hold high tick.

def cbfunc(gpio, level, tick):
   global high_tick
   if level == 0: # echo line changed from high to low.
      if high_tick is not None:
         echo = pigpio.tickDiff(high_tick, tick)
         distance = (echo / 1000000.0) * 34030 / 2
         sonar_data = Float64WithHeader()
         sonar_data.header.stamp = get_rostime()
         if abs(distance) < 2.0:
            sonar_data.float.data = distance
            pub1.publish(distance)
         else:
            sonar_data.float.data = 0.0
            pub1.publish(0.0)
         pub.publish(sonar_data)
         print("echo was {} micros long ({:.1f} cms)".format(echo, distance))
   else:
      high_tick = tick

pi = pigpio.pi() # Connect to local Pi.

pi.set_mode(TRIGGER, pigpio.OUTPUT)
pi.set_mode(ECHO, pigpio.INPUT)

cb = pi.callback(ECHO, pigpio.EITHER_EDGE, cbfunc)

def sonar():
    init_node('sonar', anonymous=True)
    rate = Rate(20) # 10hz
    while not is_shutdown():
        pi.gpio_trigger(TRIGGER, 10)
        rate.sleep()

if __name__ == '__main__':
   try:
      sonar()
   except ROSInterruptException:
      pass
   finally:
      cb.cancel() # Cancel callback.
      pi.stop() # Close connection to Pi

