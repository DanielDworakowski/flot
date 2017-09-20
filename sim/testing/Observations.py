import numpy as np
from abc import ABC, abstractmethod
#
# Class defining a position vector.
class Vec3():
    x = np.float32(0)
    y = np.float32(0)
    z = np.float32(0)
#
# Class defining a rotation with euler angles.
class RotationEuler():
    roll = np.float32(0)
    pitch = np.float32(0)
    yaw = np.float32(0)
#
# Class defining a generic set of observations.
class Observation():
    valid = False
    cameraImageU8 = None
    timestamp = None # In nanoseconds for airsim.
    cameraPosition = Vec3()
    cameraRotation = RotationEuler()
    hasCollided = False
#
# Abstract class defining the interface of an observer.
class Observer():
    #
    # Initialization.
    def __init__(self):
        pass
    #
    # Fills and returns observations about the environment.
    @abstractmethod
    def observeImpl(self, obs):
        pass
    #
    # Do the observation.
    def observe(self, obs):
        return self.observeImpl(obs)
