import numpy as np
from debug import *
from abc import abstractmethod

class Action(object):
    #
    # function to turn an array value to tangential (v_t) and angular(w) velocities
    def normalize(self):
        """ setting which action to take. action is a vector of length D, where D is the dimension of the action space. action vector can be one-hot vector but this function will take the argmax. """
        action_idx = np.argmax(self.array)
        action_sum = np.sum(self.array)

        if action_sum == 0:
            self.v_t = 0
            self.w =  0
        else:
            #
            # action_norm is 0 when the action_idx is in the middle of the act_dim, 1 when it is (act_dim-1), and -1 when action_idx is 0
            action_norm = 2.0*(1.*action_idx/(len(self.array)-1)-0.5)
            action_in_rad = action_norm*np.pi/2.
            v_t_norm = np.cos(action_in_rad)*np.sign(action_in_rad)
            w_norm = np.sin(action_in_rad)
            self.v_t = v_t_norm*self.max_v_t
            self.w = w_norm*self.max_w

    def __init__(self, array = None , v_t = None, w = None, max_v_t=0.12, max_w=0.1, z=None, isReset=False): #TODO maybe include observation
        #
        # set max vt and w
        self.max_v_t = max_v_t
        self.max_w = max_w
        self.v_t = None
        self.w = None
        self.array = None
        self.z = z
        self.isReset = isReset
        self.meta = {}
        #
        # Error checking: Too many inputs
        if (array is not None and v_t is not None and w is not None and z is not None):
            printError("Only input either array xor v_t and w")
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
                self.normalize()
            else:
                printError("wrong type for array, use []")
        elif z is None:
            printError("Incorrect format, input array XOR v_t and w")

class ActionEngine(object):
    """ Stores information to specify an action

    This class is used to communicate which action to take. The Action class can take in an one-hot vector and convert to an approaportate tangential velocty and angiular velocity. Support for velocity in z is TODO for later.

    """
    def __init__(self):
        self.v_t = 0.
        self.v_z = 0.
        self.w = 0.
        self.home_pose = [0.,0.,0.5,0.,0.,0.] # x, y, z, pitch, roll, yaw
        self.z = None
        self.isReset = False

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
            self.v_t = action.v_t
            self.w = action.w
        self.z = action.z
        self.isReset = action.isReset

    @abstractmethod
    def executeActionImpl(self, obs):
        pass

    def executeAction(self, action, obs):
        self.setAction(action)
        return self.executeActionImpl(obs)
