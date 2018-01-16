import math as m
import numpy as np
from debug import *
import time
import threading
import traceback
import SigHandler
#
# Initial class to help determine the control API.
class RobotControl(threading.Thread):
    """ Interface to control the actual blimp robot

    RobotControl class communicate with the client's ROS command publishing node
    This class is responsible for the moving the robot, as in taking actions in
    the real environment. Inherits from the Thread class to allow the
    RobotControl to run in the background.

    """

    def __init__(self):
        threading.Thread.__init__(self)
        # self.client = MultirotorClient()
        self.f = 30.
        self.running = False
        self.v_t = 0. # target tangential velocity m/s
        self.v_z = 0. # target velocity in z m/s
        self.w = 0. # target angular velocity m/s
        self.z = None

    def __enter__(self):
        """ Start the thread """
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        """ Exit the thread """
        self.running = False
        self.join()
        return False

    def executeCommand(self):
        """ Send command to publisher """
        return


    def setCommand(self, v_t_ref, w_ref, z, v_z_ref=0.):
        """ Set the desired tangential velocity, angular velocity, velocity in z """
        self.v_t = v_t_ref
        self.w = w_ref
        self.v_z = v_z_ref
        self.z = z

    def run(self):
        """ Keep going until obj destruction """
        self.running = True
        while self.running:
            self.executeCommand()

            # Rate limiting
            time.sleep(1/ self.f)
