from debug import *
from Actions import Action
import AgentBase as base
import numpy as np
import Observations as obv
import sys
import math
import random
import time

class Agent(base.AgentBase):
    PI = math.pi
    SPEED = 0.5
    DEBUG = True

    def __init__(self, conf):
        super(Agent, self).__init__()

        self.dumbAction = None
        self.mode = 0
        self.poses = []
        self.first_pose = None
        self.localCollisionCount = 0
        random.seed(time.time())

    def getActionImpl(self):
        camPos = self.obs['cameraPosition']
        camRot = self.obs['cameraRotation']
        col = self.obs['hasCollided'].val

        pose = [camPos.x, camPos.y, camPos.z, camRot.pitch, camRot.roll, camRot.yaw]
        self.poses.append(pose)
        if not self.first_pose:
            self.first_pose = pose
        pose[2] = self.first_pose[2]

        print('{}: {}'.format(camRot.getFormat(), camRot.serialize()))
        print('{}: {}'.format(camPos.getFormat(), camPos.serialize()))
        print('{}: {}'.format('self.mode', self.mode))
        print('{}: {}'.format('self.localCollisionCount', self.localCollisionCount))
        print('{}: {}'.format('col', col))
        print()
        print('{}: {}'.format('self.first_pose', self.first_pose))

        if not self.DEBUG:
            if self.mode == 0:
                self.mode = 1
                pose_len = len(self.poses)
                pose_ind = 0 if pose_len==1 else random.randint(1, pose_len-1)
                position = self.poses[pose_ind][:-1]

                angle = random.uniform(-self.PI,self.PI)
                position.append(angle)

                self.dumbAction = Action(v_t=0.0, w=0.0, reset_pose=position)
                self.poses = []

            elif self.mode == 1:
                self.mode = 2
                self.dumbAction = Action(v_t=self.SPEED, w=0.0)

            elif self.mode == 2 and col:
                self.mode = 0
                self.dumbAction = Action(v_t=0.0, w=0.0)
                self.localCollisionCount += 1

        if self.DEBUG:
            if self.mode==0:
                self.dumbAction = Action(v_t=self.SPEED, w=0.0)
                self.mode=1

            elif self.mode==1 and col:
                self.mode=0
                self.dumbAction = Action(v_t=0.0, w=0.0, reset_pose=1)

        print('')
        return self.dumbAction
