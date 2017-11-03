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
    ROT_SPEED = 5.0
    TOLERANCE = 0.1

    def __init__(self, conf):
        super(Agent, self).__init__()

        self.dumbAction = Action(v_t=self.SPEED, w=0.0)
        # self.dumbAction = None
        self.angle = None
        # self.mode = 0
        self.mode = 1
        self.localCollisionCount = 0

    def getActionImpl(self):
        camRot = self.obs['cameraRotation']
        camPos = self.obs['cameraPosition']
        col = self.obs['hasCollided'].val

        print('{}: {}'.format(camRot.getFormat(), camRot.serialize()))
        print('{}: {}'.format(camPos.getFormat(), camPos.serialize()))
        print('{}: {}'.format('self.angle', self.angle))
        print('{}: {}'.format('self.mode', self.mode))
        print('{}: {}'.format('self.localCollisionCount', self.localCollisionCount))
        print('{}: {}'.format('col', col))

        if not self.angle and self.mode==0:
            self.angle = random.uniform(-self.PI,self.PI)
            # if abs(self.angle - camRot.yaw) > self.PI:
                # self.dumbAction = Action(v_t=0.0, w=self.ROT_SPEED)
            # else:
                # self.dumbAction = Action(v_t=0.0, w=-self.ROT_SPEED)
            # # might be unused
            self.dumbAction = Action(v_t=0.0, w=self.ROT_SPEED)

        elif self.mode ==0 and  abs(self.angle - camRot.yaw)<self.TOLERANCE:
            self.dumbAction = Action(v_t=self.SPEED, w=0.0)
            self.mode = 1

        elif col and self.mode==1:
            # self.angle = self.angle + self.PI
            # if self.angle > self.PI:
                # self.angle = self.angle - 2*self.PI
            self.angle = None
            # self.mode = 2
            self.mode=0
            # self.dumbAction = Action(v_t=0.0, w=self.ROT_SPEED)
            self.dumbAction = Action(v_t=0.0, w=0.0)
            self.localCollisionCount += 1

        # elif abs(self.angle - camRot.yaw)<self.TOLERANCE and self.mode==2:
            # self.mode = 0
            # self.angle = None
            # self.dumbAction = Action(v_t=0.0, w=0.0)


        print('')
        return self.dumbAction
