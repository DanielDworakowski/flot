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

#
# Neural network agent class.

def combine(orig, filtered, rep_size):
    if len(rep_size)==2:
        rep_size.append(3)
    orignp = np.array(orig)
    filterednp = filtered.numpy()
    origsize = orignp.shape

    scaled = imresize(filterednp, rep_size, interp='bicubic')

    delta_w = origsize[1] - rep_size[1]
    delta_h = origsize[0] - rep_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    ones = np.ones(scaled.shape)
    mask= np.logical_not(np.pad(ones, ((top, bottom), (left, right), (0,0)), mode='constant'))
    padded= np.pad(scaled, ((top, bottom), (left, right), (0,0)), mode='constant')
    out = np.multiply(mask, orignp) + padded

    return out

class Agent(base.AgentBase):
    #
    # Constructor.
    def __init__(self, conf):
        super(Agent, self).__init__(conf)
        self.conf = conf
        self.toTensor = transforms.ToTensor()
        #
        # Check if cuda is available.
        self.usegpu = torch.cuda.is_available() and self.conf.usegpu
        self.ToPIL = transforms.ToPILImage()
        #
        # If you want to use visualbackprop
        self.useVisualBackProp = True
        #
        # Load the model.
        if self.conf.modelLoadPath != None and os.path.isfile(self.conf.modelLoadPath):
            if self.usegpu:
                checkpoint = torch.load(self.conf.modelLoadPath)
            else:
                print('Running the agent on CPU!')
                checkpoint = torch.load(self.conf.modelLoadPath, map_location={'cuda:0': 'cpu'})
            self.model = checkpoint['model']
            if self.useVisualBackProp:
                self.visualbackprop = VisualBackProp(self.model)
            self.nnconf = checkpoint['conf']
            self.model.load_state_dict(checkpoint['state_dict'])
            printColour('Loaded model from path: %s'%self.conf.modelLoadPath, colours.OKBLUE)
        else:
            printError('Could not load model from path: %s'%self.conf.modelLoadPath)
            raise RuntimeError

        if self.usegpu:
            self.model.cuda()

        self.model.eval()

        self.model_input_img_shape = conf.image_shape
        norm = transforms.Normalize([0,0,0], [1,1,1])
        tnn = self.nnconf.transforms
        crop = transforms.CenterCrop(self.nnconf.hyperparam.cropShape)
        for tf in tnn.transforms:
            if isinstance(tf, DataUtil.Normalize):
                norm = tf.norm
                break
        self.t = transforms.Compose([crop, transforms.ToTensor(), norm])
        #
        # Rescale transforms.
        if any(isinstance(tf, DataUtil.Rescale) for tf in tnn.transforms):
            self.model_input_img_shape = (self.nnconf.hyperparam.cropShape[0],self.nnconf.hyperparam.cropShape[1],3)
            self.t = transforms.Compose([
                transforms.ToPILImage(),
                crop,
                transforms.Resize(self.nnconf.hyperparam.image_shape),
                transforms.ToTensor(),
                norm,
            ])
        else:
            print('there is no rescale')
        #
        # Heuristic Parameters
        #
        # action dim for array
        self.action_array_dim = 11
        #
        # minimum probability of collision free to go straight
        self.straight_min_prob = 0.60
        #
        # minimum probability of collision free to stop
        self.stop_min_prob = 0.0
        #
        # min turning probabilty
        self.turn_min_prob = 0.70
        #
        #
        self.back_min_prob = 0.07
        # max vt w
        action_ = Action(np.zeros(self.action_array_dim))
        self.max_v_t = action_.max_v_t
        self.max_w = action_.max_w
    #
    # Reference to an observation
    def getActionImpl(self):
        obs = self.obs
        npimg = obs['img'].uint8Img
        # If image is available
        if npimg is not None:
            collision_free_prob = []
            softmax = torch.nn.Softmax()
            probs = None
            vbp = None
            img = self.t(npimg)
            if self.usegpu:
                img = Variable(img.unsqueeze_(0).cuda(async=True))
            else:
                img = Variable(img.unsqueeze_(0))

            classActivation = self.model(img)
            collision_free_pred = classActivation.data
            probs = self.model.getClassifications(classActivation, softmax)
            if self.useVisualBackProp:
                # tmp = self.ToPIL(self.visualbackprop.visualize(img).squeeze_())
                tmp = self.visualbackprop.visualize(img)
                combined = combine(npimg, tmp, self.model_input_img_shape)
                vbp = Image.fromarray(combined.astype('uint8'), 'RGB')

            collision_free_prob = probs.numpy()[0]

            # collision free probability
            left_prob, center_prob, right_prob = collision_free_prob[0], collision_free_prob[1], collision_free_prob[2]
            action_array = np.zeros(self.action_array_dim)
            if center_prob > self.straight_min_prob:
                #action_array[int(self.action_array_dim/2)] = 1
                #action = Action(action_array)
                action = Action(v_t=self.max_v_t,w=0)

            elif left_prob<self.stop_min_prob and center_prob<self.stop_min_prob and right_prob<self.stop_min_prob:
                action = Action(action_array)

            elif center_prob < self.back_min_prob:
                action = Action(v_t=-1.25*self.max_v_t,w=0)

            elif left_prob > right_prob and left_prob < self.turn_min_prob:
                action_array[0] = 1
                action = Action(action_array)

            elif right_prob >= left_prob and right_prob < self.turn_min_prob:
                action_array[-1] = 1
                action = Action(action_array)

            elif left_prob > right_prob:
                action = Action(v_t=left_prob*self.max_v_t,w=left_prob*self.max_w)

            else:
                action = Action(v_t=right_prob*self.max_v_t,w=right_prob*self.max_w)
            #action = Action(action_array)
            print('_____________________________________________________________________________________________________________________________________')
            print("Collsion Free Prob: left:{} center:{} right:{}".format(collision_free_prob[0], collision_free_prob[1], collision_free_prob[2]))
            print("Linear Velocity: {:.2f} Angular Velocity: {:.2f}".format(action.v_t,action.w))

            # Place the activations for visualization.
            action.meta['activations'] = probs.cpu().numpy()[0]
            action.meta['visualbackprop'] = vbp

        # Take no action when no image is available
        else:
            action = Action(v_t=0,w=0)

        # Do more stuff.
        return action
