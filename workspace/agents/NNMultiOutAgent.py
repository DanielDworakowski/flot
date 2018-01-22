from debug import *
import AgentBase as base
import Observations as obv
import os
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from Actions import Action
#
# Neural network agent class.
class Agent(base.AgentBase):
    #
    # Constructor.
    def __init__(self, conf):
        super(Agent, self).__init__(conf)
        self.conf = conf
        #
        # Check if cuda is available.
        self.usegpu = torch.cuda.is_available() and self.conf.usegpu
        #
        # Load the model.
        if self.conf.modelLoadPath != None and os.path.isfile(self.conf.modelLoadPath):
            if self.usegpu:
                checkpoint = torch.load(self.conf.modelLoadPath)
            else:
                checkpoint = torch.load(self.conf.modelLoadPath, map_location={'cuda:0': 'cpu'})
            self.model = checkpoint['model']
            self.nnconf = checkpoint['conf']
            self.model.load_state_dict(checkpoint['state_dict'])
            printColour('Loaded model from path: %s'%self.conf.modelLoadPath, colours.OKBLUE)
        else:
            printError('Could not load model from path: %s'%self.conf.modelLoadPath)
            raise RuntimeError
        if self.usegpu:
            self.model.cuda()
        self.model.eval()
        intermediateShape = self.nnconf.cropShape
        inputImgShape = self.nnconf.hyperparam.image_shape
        self.transform = transforms.Compose([
            transforms.CenterCrop(self.intermediateShape)
            transforms.Resize(self.inputImgShape)
            transforms.ToTensor()
        ])
    #
    # Reference to an observation
    def getActionImpl(self):
        obs = self.obs
        npimg = obs['img'].decompressPNG()[:,:,0:3]
        inputImg = self.transform(npimg)
        softmax = torch.nn.Softmax()
        if self.usegpu:
            img = Variable(inputImg.unsqueeze_(0).cuda())
        else:
            img = Variable(inputImg.unsqueeze_(0))
        collision_free_pred = self.model(img).data
        # collision_free_prob.append(softmax(collision_free_pred)[0,1].data.cpu().numpy(0))
        #
        # collision free probability
        # left_prob, center_prob, right_prob = [left_prob[0], center_prob[0], right_prob[0]]
        # action_array = np.zeros(self.action_array_dim)
        # if center_prob > self.straight_min_prob:
        #     action_array[int(self.action_array_dim/2)] = 1
        #     action = Action(action_array)
        #
        # elif left_prob<self.stop_min_prob and center_prob<self.stop_min_prob and right_prob<self.stop_min_prob:
        #     action = Action(action_array)
        #
        # elif left_prob > right_prob and left_prob < self.turn_min_prob:
        #     action_array[0] = 1
        #     action = Action(action_array)
        #
        # elif right_prob >= left_prob and right_prob < self.turn_min_prob:
        #     action_array[-1] = 1
        #     action = Action(action_array)
        #
        # elif left_prob > right_prob:
        #     action = Action(v_t=left_prob*self.max_v_t,w=left_prob*self.max_w)
        #
        # else:
        #     action = Action(v_t=right_prob*self.max_v_t,w=right_prob*self.max_w)
        # # action = Action(action_array)
        # print('_____________________________________________________________________________________________________________________________________')
        # print("Collsion Free Prob: left:{} center:{} right:{}".format(collision_free_prob[0], collision_free_prob[1], collision_free_prob[2]))
        # print("Linear Velocity: {} Angular Velocity: {}".format(action.v_t,action.w))

        # Do more stuff.
        action = Action(np.array(3))
        return action
