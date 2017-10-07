import Observations as obv
from debug import *
from abc import ABC, abstractmethod
#
# Base class for agents for common interface.
class AgentBase():
    obs = None
    #
    # Interface to give an observation.
    def giveObservation(self, obs):
        self.obs = obs
    #
    # Interface to obtain an action.
    def getAction(self):
        return self.getActionImpl()
    #
    # Implementation of how an action is retrieved.
    @abstractmethod
    def getActionImpl(self):
        pass
