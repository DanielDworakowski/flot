import math as m
from PythonClient import *
from debug import *
import time
import threading
#
# Initial class to help determine the control API.
class AirControl():#threading.Thread):
    #
    # Constructor.
    def __init__(self):
        #
        # Member variables.
        # threading.Thread.__init__(self)
        self.client = AirSimClient()
        self.home = None
        self.f = 30.
        self.running = False
        #
        # Begin construction.
        if not self.client.confirmConnection():
            print("Connection failed.")
        #
        # Enabling twice might break stuff.
        if not self.client.isApiControlEnabled():
            self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.home = self.client.getPosition()
        #
        # Bring off of the ground.
        if self.home.z_val > -5:
            self.home.z_val -= 40
            self.client.moveToZ(self.home.z_val, 10)
    # #
    # # Start the thread.
    # def __enter__(self):
    #     # self.start()
    #     pass
    # #
    # # Exit the thread.
    # def __exit__(self ,type, value, traceback):
    #     self.running = False
    #     # self.join()
    #     return True
    #
    # Send a single command for path following.
    def followPathStep(self, v_t, radius):
        w_deg = v_t / radius * 180 / m.pi
        yaw = self.client.getRollPitchYaw()[2]
        v_x = v_t * m.cos(yaw)
        v_y = v_t * m.sin(yaw)
        self.client.moveByVelocity(v_x, v_y, 0, 1, DrivetrainType.ForwardOnly, YawMode(True, w_deg))
    #
    # Follow a path based on a radius
    def followPathSync(self, t = 1, v_t = 1, radius = m.inf):
        steps = int(self.f * t)
        for x in range(steps):
            self.followPathStep(v_t, radius)
            time.sleep(1 / self.f)
    #
    # The thread run function.
    def run(self):
        #
        # Keep going until obj destruction
        self.running = True
        while self.running:
            self.followPathStep(1,1)
            time.sleep(1/ self.f)
