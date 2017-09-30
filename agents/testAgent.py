from debug import *
import AgentBase as base
printError('Import actions here')
import Observations as obv

class Agent(base.AgentBase):
    #
    # Reference to an observation
    def getActionImpl(self):
        printError('Implement something here')
        return {'testAction': 'test'}
