from Actions import ActionEngine, Action
import numpy as np
from debug import *

class StaticAction(Action):
    def __init__(self, array, v_t, w): #add teleport thing
        Action.__init__(self, array, v_t, w)

class StaticActionEngine(ActionEngine):
    def __init__(self):
        ActionEngine.__init__(self)

    def executeActionImpl(self, obs):
        print(self.v_t)
        print(self.w)

        return True
