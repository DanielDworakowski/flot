import numpy as np
from abc import ABC, abstractmethod
#
# Class defining a position vector.
class Vec3():
    x = np.float32(0)
    y = np.float32(0)
    z = np.float32(0)
    #
    # Order in string format.
    @staticmethod
    def getFormat():
        return ('x[m], y[m], z[m]')
    #
    # Get the values into string format.
    def toString(self):
        return ('%s, %s, %s')%(self.x, self.y, self.z)
#
# Class defining a rotation with euler angles.
class RotationEuler():
    roll = np.float32(0)
    pitch = np.float32(0)
    yaw = np.float32(0)
    #
    # Order in string format.
    @staticmethod
    def getFormat():
        return ('roll[deg], pitch[deg], yaw[deg]')
    #
    # Get the values into string format.
    def toString(self):
        return ('%s, %s, %s')%(self.roll, self.pitch, self.yaw)
#
# Class defining a generic set of observations.
class Observation():
    valid = False
    cameraImageU8 = None
    cameraImageCompressed = None
    timestamp = None # In nanoseconds for airsim.
    cameraPosition = Vec3()
    cameraRotation = RotationEuler()
    hasCollided = False
#
# Abstract class defining the interface of an observer.
class Observer():
    #
    # Initialization.
    def __init__(self, obsDir = ''):
        self.obsSir = obsDir
        self.obsCsv = None
        self.obs = Observation()
    #
    # Open observation file.
    def __enter__(self):
        self.obsCsv = open(obsDir + 'labels.csv', 'wb')
        return self
    #
    # Close the observation file.
    def __exit__(self, type, value, traceback):
        if self.obsCsv:
            self.obsCsv.close()
            self.obsCsv = None
        return False
    #
    # Fills and returns observations about the environment.
    @abstractmethod
    def observeImpl(self):
        pass
    #
    # Do the observation.
    def observe(self):
        return self.observeImpl(self.obs)
    #
    # Serialize the data.
    def serialize(self):
        pass
