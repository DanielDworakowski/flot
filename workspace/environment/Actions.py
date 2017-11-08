import numpy as np
from debug import *
from abc import ABC, abstractmethod

class Action():
    def __init__(self, array = None , v_t = None, w = None, max_v_t=1.0, max_w=2.0, z=None): #TODO maybe include observation
        #
        # set max vt and w
        self.max_v_t = max_v_t
        self.max_w = max_w
        self.v_t = None
        self.w = None
        self.array = None
        self.z = z
        #
        # Error checking: Too many inputs
        if (array is not None and v_t is not None and w is not None and z is not None):
            printError("Only input either array xor v_t and w")
        #
        # function to turn an array value to tangential (v_t) and angular(w) velocities
        def normalize(self):
            """ setting which action to take. action is a vector of length D, where D is the dimension of the action space. action vector can be one-hot vector but this function will take the argmax. """
            action_idx = np.argmax(self.array)

            if action_idx == 0:
                self.v_t = 0
                self.w =  0
            else:
                #
                # action_norm is 0 when the action_idx is in the middle of the act_dim, 1 when it is (act_dim-1), and -1 when action_idx is 0
                action_norm = 2.0*(1.*action_idx/(len(self.array)-1)-0.5)
                action_in_rad = action_norm*np.pi/2.
                v_t_norm = np.cos(action_in_rad)
                w_norm = np.sin(action_in_rad)
                self.v_t = v_t_norm
                self.w = w_norm
        #
        #Check if v_t and w are the ONLY inputs
        if ( v_t is not None and w is not None and array is None):
            #
            #Check if the types are valid
            if isinstance(v_t, int): v_t = float(v_t)
            if isinstance(w, int): w = float(w)

            if ( isinstance(v_t, float) and isinstance(w, float)):
                self.v_t = v_t
                self.w = w
            else:
                printError("wrong type for v_t and w, use float")
        #
        #Check if array is the ONLY input
        elif (array is not None and v_t is None and w is None):
            #
            #Check if the types are valid
            if ( isinstance(array, list) or isinstance(array, np.ndarray)):
                self.array = array
                normalize(self)
            else:
                printError("wrong type for array, use []")
        elif z is None:
            printError("Incorrect format, input array XOR v_t and w")



class ActionEngine():
    """ Stores information to specify an action

    This class is used to communicate which action to take. The Action class can take in an one-hot vector and convert to an approaportate tangential velocty and angiular velocity. Support for velocity in z is TODO for later.

    """
    def __init__(self):
        self.v_t = 0.
        self.v_z = 0.
        self.w = 0.
        self.home_pose = [0.,0.,0.5,0.,0.,0.] # x, y, z, pitch, roll, yaw
        self.z = None

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
    #
    # saturator function
    def setAction(self, action):
        if action.v_t is not None and action.w is not None:
            self.v_t = action.max_v_t*action.v_t
            self.w = action.max_w*action.w
        self.z = action.z

        # print(self.v_t, "   ", self.w)

    @abstractmethod
    def executeActionImpl(self, obs):
        pass

    def executeAction(self, action, obs):
        self.setAction(action)
        return self.executeActionImpl(obs)
