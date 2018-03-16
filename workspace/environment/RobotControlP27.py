import time
import rospy
import atexit
import threading
import traceback
import math as m
import SigHandler
import numpy as np
from debug import *
from subprocess import Popen
from std_msgs.msg import Float64
#
# Initial class to help determine the control API.
class RobotControl(object):
    """ Interface to control the actual blimp robot

    RobotControl class communicate with the client's ROS command publishing node
    This class is responsible for the moving the robot, as in taking actions in
    the real environment. Inherits from the Thread class to allow the
    RobotControl to run in the background.

    """

    def __init__(self):
        self.running = False
        self.v_t = 0.   # target tangential velocity m/s
        self.v_z = 0.   # target velocity in z m/s
        self.w = 0.     # target angular velocity m/s
        self.z = None
        #
        # Initialize ROS.
        rospy.init_node('blimp_cmd_pub', anonymous=True)
        self.vt_pub = rospy.Publisher('/blimp_vt', Float64, queue_size=1)
        self.alt_pub = rospy.Publisher('/blimp_alt', Float64, queue_size=1)
        self.w_pub = rospy.Publisher('/blimp_w', Float64, queue_size=1)
        self.msg = Float64()

    def __enter__(self):
        """ Start the thread """
        return self

    def __exit__(self, type, value, traceback):
        """ Exit the thread """
        # Reset RobotCommands attributes (v_t, v_z, w) to None, if possible
        try:
            self.setCommand(0,0,0)
        except:
            pass

        return False

    def _executeCommand(self):
        """ Send command to publisher """

        if self.v_t is None:
            self.v_t = 0.3
        if self.w is None:
            self.w = 0.1

        self.msg.data = float(self.v_t)
        self.vt_pub.publish(self.msg)

        self.msg.data = float(self.v_z)
        self.alt_pub.publish(self.msg)

        self.msg.data = float(self.w)
        self.w_pub.publish(self.msg)

    def setCommand(self, v_t_ref, w_ref, z=0., v_z_ref=1):
        """ Set the desired tangential velocity, angular velocity, velocity in z """
        self.v_t = v_t_ref
        self.w = w_ref
        self.v_z = v_z_ref
        self.z = z
        # print('setting commands')
        self._executeCommand()
