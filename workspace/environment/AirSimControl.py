import math as m
import numpy as np
from AirSimClient import *
from debug import *
import time
import threading
import traceback
import SigHandler
#
# Initial class to help determine the control API.
class AirSimControl(threading.Thread):
    """ Interface to control the robot in AirSim

    AirSimControl class, communicates with AirSim via its PythonClient. This class is responsible for the moving the robot, as in taking actions in the AirSim environment. Inherits from the Thread class to allow the AirSimControl to run in the background.

    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.client = MultirotorClient()
        self.f = 30.
        self.running = False
        self.set_pose_request = False
        self.set_pose_position = Vector3r(0,0,0)
        self.set_pose_quaternion = AirSimClientBase.toQuaternion(0,0,0)
        self.v_t = 0. # target tangential velocity m/s
        self.v_z = 0. # target velocity in z m/s
        self.w = 0. # target angular velocity m/s
        self.z = None
        #
        # Begin construction.
        self.client.confirmConnection()
        #
        # Enabling twice might break stuff.
        if not self.client.isApiControlEnabled():
            self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoff()
        print("ROBOT READY")

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
        """ Send a single command for path following """
        if self.z is None: #z dne
            yaw = self.client.getRollPitchYaw()[2]
            v_x = self.v_t * m.cos(yaw)
            v_y = self.v_t * m.sin(yaw)
            success = self.client.moveByVelocity(v_x, v_y, self.v_z, 1, DrivetrainType.ForwardOnly, YawMode(True, self.w))
            if not success:
                print('Control: velocity command failed.')
        elif self.v_t is not None and self.w is not None: #z exists and v,w exists too
            yaw = self.client.getRollPitchYaw()[2]
            v_x = self.v_t * m.cos(yaw)
            v_y = self.v_t * m.sin(yaw)
            z = self.z
            success = self.client.moveByVelocityZ(v_x, v_y, z, 1, DrivetrainType.ForwardOnly, YawMode(True, self.w))
            if not success:
                print('Control: velocity command failed.')
        else:
            z = self.z
            success = self.client.moveToZ(z, 10)


    def setCommand(self, v_t_ref, w_ref, z, v_z_ref=0.):
        """ Set the desired tangential velocity, angular velocity, velocity in z """
        self.v_t = v_t_ref
        self.w = w_ref
        self.v_z = v_z_ref
        self.z = z

    def setPose(self, pose):
        x, y, z, pitch, roll, yaw = pose
        self.set_pose_request = True
        self.set_pose_position = Vector3r(x, y, -z)
        # self.pose = pose
        self.set_pose_quaternion = AirSimClientBase.toQuaternion(pitch, roll, yaw)

    def run(self):
        """ Keep going until obj destruction """
        self.running = True
        while self.running:
            if self.set_pose_request:
                # newPose = Pose(self.set_pose_position, self.set_pose_quaternion)
                # self.client.simSetPose(newPose, True)
                # self.client.reset()
                # self.client.enableApiControl(True)
                # self.client.armDisarm(True)
                # self.client.takeoff()

                # self.client.moveToPosition(self.pose[0],self.pose[1],self.pose[2],20)
                # self.client.rotateToYaw(self.pose[-1], margin=0.05)
                newPose = Pose(self.set_pose_position, self.set_pose_quaternion)
                # self.client.simSetPose(newPose, True)
                self.client.reset()
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                # self.client.takeoff()

                self.set_pose_request = False
            self.executeCommand()
            time.sleep(1/ self.f)
