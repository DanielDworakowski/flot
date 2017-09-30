import AirSimControl
from Actions import ActionEngine
import numpy as np 
from debug import *

class AirSimActionEngine(ActionEngine):
	def __init__(self, act_dim=3, sim=True, max_v_t=2.0, max_w=2.0):
		ActionEngine.__init__(self, act_dim, sim, max_v_t, max_w)
		self.asc = AirSimControl.AirSimControl()

	def reset(self, pose=None):
		if pose is not None:
			self.home_pose = pose
		self.asc.setPose(self.home_pose)

	def executeActionImpl(self):
		success = True
		try:
			self.asc.setCommand(self.v_t, self.w)
		except:
			success = False
			printError("Failed to execute comman in AirSim!")
		return success
