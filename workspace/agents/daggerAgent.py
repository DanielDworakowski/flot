from debug import *
import AgentBase as base
import Observations as obv
import os
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from Actions import Action

import sys
import math
import random
import time

def angdiff(t,s):
    return math.atan2(math.sin(t-s), math.cos(t-s))
#
# Neural network agent class.
class Agent(base.AgentBase):
    PI = math.pi
    SPEED = 1.0
    ROT_SPEED = 20.0
    TOLERANCE = 0.05
    #
    # Constructor.
    def __init__(self, conf):
        super(Agent, self).__init__(conf)
        self.conf = conf
        self.toTensor = transforms.ToTensor()
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
        self.model_input_img_shape = conf.image_shape
        #
        # Heuristic Parameters
        #
        # action dim for array
        self.action_array_dim = 11
        #
        # minimum probability of collision free to go straight
        self.straight_min_prob = 0.80
        #
        # minimum probability of collision free to stop
        self.stop_min_prob = 0.0
        #
        # min turning probabilty
        self.turn_min_prob = 0.70
        #
        # max vt w
        action_ = Action(np.zeros(self.action_array_dim))
        self.max_v_t = action_.max_v_t
        self.max_w = action_.max_w

        self.mode = 0
        self.angle = None
        self.last_pose = None
        self.last_time = None
        self.still_counter = 0
        random.seed(time.time())
    #
    # Crop image into three sections, left center right.
    def cropImageToThree(self, npimg):
        #
        # shape of the image to split
        img_h, img_w, img_c = npimg.shape
        model_img_h, model_img_w, model_img_c = self.model_input_img_shape
        #
        # check if the image to crop is large enough
        if self.model_input_img_shape[0] > img_h or self.model_input_img_shape[1] > img_w:
            printError("The image cannot be cropped because the image is too small. Model Image Shape:"+str(self.model_input_img_shape)," Image Given:"+str(npimg.shape))
            raise RuntimeError
        #
        # calculate the idx to start
        h_0 = int((img_h - model_img_h)/2)
        w_0 = 0
        w_1 = int((img_w - model_img_w)/2)
        w_2 = img_w - model_img_w
        #
        # cropping the image
        left_img = npimg[h_0:h_0+model_img_h,w_0:w_0+model_img_w,:]
        center_img = npimg[h_0:h_0+model_img_h,w_1:w_1+model_img_w,:]
        right_img = npimg[h_0:h_0+model_img_h,w_2:w_2+model_img_w,:]

        return [left_img, center_img, right_img]

    def smartAction(self):
        npimg = self.obs['img'].decompressPNG()[:,:,0:3]
        cropped_imgs = self.cropImageToThree(npimg)
        collision_free_prob = []
        softmax = torch.nn.Softmax()
        probs = None
        for idx, cropped_img in enumerate(cropped_imgs):
            if self.usegpu:
                img = Variable(self.toTensor(cropped_img).unsqueeze_(0).cuda())
            else:
                img = Variable(self.toTensor(cropped_img).unsqueeze_(0))
            classActivation = self.model(img)
            collision_free_pred = classActivation.data
            if idx == 1:
                probs = self.model.getClassifications(classActivation, softmax)
            collision_free_prob.append(softmax(classActivation)[0,1].data.cpu().numpy())
        #
        # collision free probability
        left_prob, center_prob, right_prob = collision_free_prob
        left_prob, center_prob, right_prob = [left_prob[0], center_prob[0], right_prob[0]]
        action_array = np.zeros(self.action_array_dim)
        if center_prob > self.straight_min_prob:
            printFrame()
            action_array[int(self.action_array_dim/2)] = 1
            action = Action(action_array)

        elif left_prob<self.stop_min_prob and center_prob<self.stop_min_prob and right_prob<self.stop_min_prob:
            printFrame()
            action = Action(action_array)

        elif right_prob < self.turn_min_prob and left_prob < self.turn_min_prob:
            printFrame()
            action_array[0] = 1
            action = Action(action_array)

        elif left_prob > right_prob and left_prob < self.turn_min_prob:
            printFrame()
            action_array[0] = 1
            action = Action(action_array)

        elif right_prob >= left_prob and right_prob < self.turn_min_prob:
            printFrame()
            action_array[-1] = 1
            action = Action(action_array)

        elif left_prob > right_prob:
            printFrame()
            action = Action(v_t=left_prob*self.max_v_t,w=-left_prob*self.max_w)

        else:
            printFrame()
            action = Action(v_t=right_prob*self.max_v_t,w=right_prob*self.max_w)
        # action = Action(action_array)
        print('_____________________________________________________________________________________________________________________________________')
        print("Collsion Free Prob: left:{} center:{} right:{}".format(collision_free_prob[0], collision_free_prob[1], collision_free_prob[2]))
        print("Linear Velocity: {} Angular Velocity: {}".format(action.v_t,action.w))
        #
        # Place the activations for visualization.
        action.meta['activations'] = probs.cpu().numpy()[0]
        #
        # Do more stuff.
        return action
    #
    # Reference to an observation
    def getActionImpl(self):
        col = self.obs['hasCollided'].val
        camPos = self.obs['cameraPosition']
        camRot = self.obs['cameraRotation']
        pose = [camPos.x, camPos.y, camPos.z, \
                camRot.pitch, camRot.roll, camRot.yaw]

        if self.last_pose != pose:
            self.last_pose = pose
            self.still_counter = 0
        else:
            self.still_counter += 1

        print('{}: {}'.format('self.mode', self.mode))
        print('{}: {}'.format('pose', pose))
        print('{}: {}'.format('col', col))
        print()

        if self.angle:
            diff = angdiff(self.angle, camRot.yaw)

        # if self.still_counter > 15:
        #     print('quit\n')
        #     quit()
        #
        # el
        if self.mode == 0 and self.angle is None:
            self.angle = random.uniform(-self.PI,self.PI)
            action = Action(v_t=0.0, w=0.0)

        elif self.mode == 0 and abs(diff) > self.TOLERANCE:
            speed = self.ROT_SPEED*diff
            action = Action(v_t=0.0, w=speed)

        elif self.mode == 0 and abs(diff) < self.TOLERANCE:
            action = Action(v_t=0.0, w=0.0)
            self.mode = 1

        elif self.mode == 1 and not col:
            action = self.smartAction()
            ############################

        elif self.mode == 1 and col:
            self.flight_duration = random.uniform(0.5,2.5)
            self.mode = 2

            self.angle = None
            action = Action(v_t=0.0, w=0.0)
            self.last_time = time.time()

        elif self.mode == 2 and time.time()-self.last_time <= 1.5:
            action = Action(v_t=0.0, w=0.0)

        elif self.mode == 2 and time.time()-self.last_time > 1.5:
            self.mode = 3
            action = Action(v_t=-self.SPEED, w=0.0)
            self.last_time = time.time()

        elif self.mode == 3 and time.time()-self.last_time <= self.flight_duration:
            action = Action(v_t=-self.SPEED, w=0.0)

        elif self.mode == 3 and time.time()-self.last_time > self.flight_duration:
            self.mode = 0
            action = Action(v_t=0.0, w=0.0)
            self.last_time = None
        action.z = -1.45

        return action
