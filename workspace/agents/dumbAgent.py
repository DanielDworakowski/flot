from debug import *
from Actions import Action
import AgentBase as base
import numpy as np
import Observations as obv
import sys
import math
import random
import time

def angdiff(t,s):
    return math.atan2(math.sin(t-s), math.cos(t-s))

class Agent(base.AgentBase):
    PI = math.pi
    SPEED = 0.1
    ROT_SPEED = 0.1
    TOLERANCE = 0.05
    DEBUG = False

    def __init__(self, conf):
        super(Agent, self).__init__()

        self.dumbAction = None
        self.angle = None
        self.mode = 0
        self.last_time = None

        self.last_pose = None
        self.still_counter = 0

        self.flight_duration = None
        random.seed(time.time())

    def debugAction(self):
        if self.mode==0:
            self.dumbAction = Action(v_t=self.SPEED, w=0.0)
            self.mode=1

        # elif self.mode==1 and col:
        #     self.mode=0
        #     quit()
            # self.dumbAction = Action(v_t=0.0, w=0.0, isReset=True)

    def getActionImpl(self):
        camPos = self.obs['cameraPosition']
        camRot = self.obs['cameraRotation']

        col = self.obs['hasCollided'].val

        pose = [camPos.x, camPos.y, camPos.z, \
                camRot.pitch, camRot.roll, camRot.yaw]

        if self.last_pose != pose:
            self.last_pose = pose
            self.still_counter = 0
        else:
            self.still_counter += 1

        print('{}: {}'.format('pose', pose))
        print('{}: {}'.format('self.angle', self.angle))
        print('{}: {}'.format('self.mode', self.mode))
        print('{}: {}'.format('self.still_counter', self.still_counter))
        print('{}: {}'.format('col', col))
        print()

        if self.angle:
            diff = angdiff(self.angle, camRot.yaw)
            diff = 0

        if not self.DEBUG:

            # if self.still_counter > 15:
            #     quit()
            #
            # el
            if self.mode == 0 and self.angle is None:
                self.angle = random.uniform(-0.1,0.1)
                self.dumbAction = Action(v_t=0.0, w=0.0)

            elif self.mode ==0 and abs(diff) > self.TOLERANCE:
                speed = self.ROT_SPEED*diff
                # if angdiff(self.angle, camRot.yaw) < 0:
                    # speed *= -1
                self.dumbAction = Action(v_t=0.0, w=speed)

            elif self.mode ==0 and abs(diff) < self.TOLERANCE:
                self.dumbAction = Action(v_t=self.SPEED, w=self.angle)
                self.mode = 1
                self.last_time = time.time()

            elif self.mode==1 and time.time()-self.last_time > 5:
                duration = random.uniform(0,time.time()-self.last_time)
                self.flight_duration = max(duration, 0.5)
                self.mode = 2

                self.angle = None
                self.dumbAction = Action(v_t=0.0, w=0.0)
                self.last_time = time.time()

            elif self.mode==2 and time.time()-self.last_time > 1.5:
                self.mode = 3
                self.dumbAction = Action(v_t=-self.SPEED, w=0.0)
                self.last_time = time.time()

            elif self.mode==3 and time.time()-self.last_time > 5:
                self.mode = 0
                self.dumbAction = Action(v_t=0.0, w=0.0)
                self.last_time = None

        if self.DEBUG:
            self.debugAction()

        print('Linear %f, Angular %f'%(self.dumbAction.v_t, self.dumbAction.w))

        self.dumbAction.meta['visualbackprop'] = False
        return self.dumbAction
