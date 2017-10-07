from debug import *
import AgentBase as base
import numpy as np
import Observations as obv

class Agent(base.AgentBase):
    #
    # Reference to an observation
    def getActionImpl(self):
        if self.obs == None:
        	printError("Agent's Observation is None, make sure to giveObservation to Agent before calling getAction")
        else:
        	# return np.array([1,0,0])
        	# return np.array([0,0,1])
        	return np.array([0,1,0])
        	# return np.random.randn(3)