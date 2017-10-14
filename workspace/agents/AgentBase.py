import Observations as obv
from debug import *
from abc import ABC, abstractmethod
#
# Base class for agents for common interface.
class AgentBase():
    obs = None
    conf = None
    #
    # Interface to give an observation.
    def giveObservation(self, obs):
        self.obs = obs
    #
    # Interface to obtain an action.
    def getAction(self):
        if self.obs == None:
        	printError("Agent's Observation is None, make sure to giveObservation to Agent before calling getAction")
        else:
            return self.getActionImpl()
    #
    # Implementation of how an action is retrieved.
    @abstractmethod
    def getActionImpl(self):
        pass
    #
    # Implementation of initiailization.
    def __init__(self, conf = None):
        self.conf = conf
