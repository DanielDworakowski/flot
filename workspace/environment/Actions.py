import numpy as np
from debug import *
from abc import ABC, abstractmethod

class ActionEngine():
    """ Stores information to specify an action 

    This class is used to communicate which action to take. The Action class can take in an one-hot vector and convert to an approaportate tangential velocty and angiular velocity. Support for velocity in z is TODO for later.

    """
    def __init__(self, act_dim=3, max_v_t=1.0, max_w=2.0):
        self.act_dim = act_dim
        self.max_v_t = max_v_t
        self.max_w = max_w
        self.v_t = 0.
        self.v_z = 0.
        self.w = 0.
        self.home_pose = np.zeros(6) # x, y, z, pitch, roll, yaw

    @abstractmethod    
    def __enterImpl__(self):
        pass

    @abstractmethod
    def __exitImpl__(self, type, value, traceback):
        pass

    def __enter__(self):
        self.__enterImpl__()
        return self

    def __exit__(self, type, value, traceback):
        self.__exitImpl__(type, value, traceback)
        return False

    def setAction(self, action):
        """ setting which action to take. action is a vector of length D, where D is the dimension of the action space. action vector can be one-hot vector but this function will take the argmax. """

        action_idx = np.argmax(action)
        #
        # action_norm is 0 when the action_idx is in the middle of the act_dim, 1 when it is (act_dim-1), and -1 when action_idx is 0
        action_norm = 2.0*(1.*action_idx/(self.act_dim-1)-0.5)
        action_in_rad = action_norm*np.pi/2.
        v_t_norm = np.cos(action_in_rad)
        w_norm = np.sin(action_in_rad)
        self.v_t = self.max_v_t*v_t_norm
        self.w = self.max_w*w_norm

    @abstractmethod
    def executeActionImpl(self):
        pass

    def executeAction(self, action):
        self.setAction(action)
        return self.executeActionImpl()
