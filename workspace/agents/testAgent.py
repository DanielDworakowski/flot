from debug import *
import AgentBase as base
import numpy as np
import Observations as obv
import sys

class Agent(base.AgentBase):
    #
    # Initialize.
    def __init__(self):
        super(Agent, self).__init__()
        self.numObs = 17000
        self.obsCount = 0
        self.shift = 0
        print("Running Agent to collect {} Observations".format(self.numObs))
    #
    # Reference to an observation
    def getActionImpl(self):
        if self.obs == None:
            printError("Agent's Observation is None, make sure to giveObservation to Agent before calling getAction")
        else:
            if self.numObs == self.obsCount:
                print("Data Collection Complete")
                sys.exit()
            self.obsCount += 1
            if self.obsCount % 50 == 0:
                print("{} Data Collected".format(self.obsCount))
                self.shift = int(np.random.normal())*5
            # return np.array([1,0,0])
            # return np.array([0,0,1])
            # return np.array([0,1,0])
            # return np.array([0,0,0,0,0,0,0,1,0,0,0])
            return np.roll(abs(np.random.randn(11)*np.array([3,6,12,25,50,80,50,25,12,6,3])),self.shift)