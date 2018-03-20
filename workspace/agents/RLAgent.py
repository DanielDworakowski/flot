import os
import torch
import numpy as np
from debug import *
import AgentBase as base
from Actions import Action
import Observations as obv
from torchvision import transforms
import nn.util.DataUtil as DataUtil
from torch.autograd import Variable
from util.visualBackProp import VisualBackProp
from scipy.misc import imresize
from PIL import Image
from rl.algorithms.utils.PolicyNetwork import A2CPolicyNetwork

class Agent(base.AgentBase):
    #
    # Constructor.
    def __init__(self, conf):
        super(Agent, self).__init__(conf)
        self.conf = conf
        self.policy_network = A2CPolicyNetwork(None, 2, None, 4)

        #
        # Load the model.
        if self.conf.modelLoadPath != None and os.path.isfile(self.conf.modelLoadPath):
            self.policy_network.load_state_dict(torch.load(self.conf.modelLoadPath))
        else:
            printError('Could not load model from path: %s'%self.conf.modelLoadPath)
            raise RuntimeError

        self.max_v_t = action_.max_v_t
        self.max_w = action_.max_w

    #
    # Reference to an observation
    def getActionImpl(self):
        obs = self.obs
        npimg = obs['img'].uint8Img
        # If image is available
        if npimg is not None:
            action = self.policy_network.compute(observation)
            action = np.clip(action, -1, 1)
            v_ref, w_ref = action[0], action[1]
            action = Action(v_t=-v_ref*self.max_v_t,w=w_ref*self.max_w)
            print("Linear Velocity: {:.2f} Angular Velocity: {:.2f}".format(action.v_t,action.w))

        # Take no action when no image is available
        else:
            action = Action(v_t=0,w=0)

        # Do more stuff.
        return action
