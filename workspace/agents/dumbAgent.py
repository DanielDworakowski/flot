from debug import *
from Actions import Action
import AgentBase as base
import numpy as np
import Observations as obv
import sys
import math
import random

class Agent(base.AgentBase):
    PI = math.pi
    SPEED = 5.0
    ROT_SPEED = 30.0

    def __init__(self, conf):
        super(Agent, self).__init__()

        self.dumbAction = None
        self.angle = None
        self.move = False
        self.localCollisionCount = 0
        self.currpose = None

    def getActionImpl(self):
        camRot = self.obs['cameraRotation']
        camPos = self.obs['cameraPosition']

        print('{}: {}'.format(camRot.getFormat(), camRot.serialize()))
        print('{}: {}'.format(camPos.getFormat(), camPos.serialize()))
        print('{}: {}'.format('self.angle', self.angle))
        print('{}: {}'.format('self.move', self.move))
        print('{}: {}'.format('self.localCollisionCount', self.localCollisionCount))
        print('{}: {}'.format('self.currpose', self.currpose))

        if not self.angle:
            self.angle = random.uniform(-self.PI,self.PI)
            # if abs(self.angle - camRot.yaw) > self.PI:
                # self.dumbAction = Action(v_t=0.0, w=self.ROT_SPEED)
            # else:
                # self.dumbAction = Action(v_t=0.0, w=-self.ROT_SPEED)
            # might be unused
            self.dumbAction = Action(v_t=0.0, w=self.ROT_SPEED)
            # return self.dumbAction

        if not self.move and abs(self.angle - camRot.yaw)<0.05:
            self.dumbAction = Action(v_t=self.SPEED, w=0.0)
            self.move = True
            # return self.dumbAction

        col = self.obs['hasCollided'].val
        print('{}: {}'.format('col', col))

        if col:
            self.angle = None
            self.move = False
            self.dumbAction = Action(v_t=0.0, w=0.0, reset_pose=1)
            self.localCollisionCount += 1

        print('')
        return self.dumbAction
