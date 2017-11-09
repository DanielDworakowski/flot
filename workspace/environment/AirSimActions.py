import AirSimControl
from Actions import ActionEngine, Action
import numpy as np
from debug import *

class AirSimAction(Action):
    def __init__(self, array, v_t, w): #add teleport thing
        Action.__init__(self, array, v_t, w)


class AirSimActionEngine(ActionEngine):
    def __init__(self):
        ActionEngine.__init__(self)
        self.asc = AirSimControl.AirSimControl()

    def __enterImpl__(self):
        self.asc.__enter__()

    def __exitImpl__(self, type, value, traceback):
        self.asc.__exit__(type, value, traceback)

    def reset(self, pose=None):
        if pose:
            self.asc.setPose(pose)
        else:
            self.asc.setPose(self.home_pose)

    def executeActionImpl(self, obs):
        success = True
        try:
            if self.isReset:
                self.reset()
            # fix/reformat later
            # if obs.serializable['hasCollided'].val:
                # self.reset()
            self.asc.setCommand(self.v_t, self.w, self.z)
            # print(self.v_t, self.w)
        except Exception as e:
            success = False
            printError("Failed to execute command in AirSim!")
            if hasattr(e, 'message'):
                printError(e.message)
            else:
                printError(e)


        return success
