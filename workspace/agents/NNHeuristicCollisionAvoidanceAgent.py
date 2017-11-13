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
        self.toTensor = transforms.ToTensor()
        #
        # Load the model.
        if self.conf.modelLoadPath != None and os.path.isfile(self.conf.modelLoadPath):
            checkpoint = torch.load(self.conf.modelLoadPath)
            self.model = checkpoint['model']
            self.model.load_state_dict(checkpoint['state_dict'])
            printColour('Loaded model from path: %s'%self.conf.modelLoadPath, colours.OKBLUE)
        else:
            printError('Could not load model from path: %s'%self.conf.modelLoadPath)
            raise RuntimeError
        #
        # Check if cuda is available.
        self.usegpu = torch.cuda.is_available() and self.conf.usegpu
        if self.usegpu:
            self.model.cuda()
        else:
            printError('This computer does not have CUDA, stuff may not work')
            raise RuntimeError
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
    #
    # Reference to an observation
    def getActionImpl(self):
        obs = self.obs
        npimg = obs['img'].decompressPNG()[:,:,0:3]
        cropped_imgs = self.cropImageToThree(npimg)
        collision_free_prob = []
        softmax = torch.nn.Softmax()
        for cropped_img in cropped_imgs:
            img = Variable(self.toTensor(cropped_img).unsqueeze_(0).cuda())
            collision_free_pred = self.model(img).data
            collision_free_prob.append(softmax(collision_free_pred)[0,1].data.cpu().numpy(0))
        #
        # collision free probability
        left_prob, center_prob, right_prob = collision_free_prob
        left_prob, center_prob, right_prob = [left_prob[0], center_prob[0], right_prob[0]]
        action_array = np.zeros(self.action_array_dim)
        if center_prob > self.straight_min_prob:
            action_array[int(self.action_array_dim/2)] = 1
            action = Action(action_array)

        elif left_prob<self.stop_min_prob and center_prob<self.stop_min_prob and right_prob<self.stop_min_prob:
            action = Action(action_array)

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
        # action = Action(action_array)
        print('_____________________________________________________________________________________________________________________________________')
        print("Collsion Free Prob: left:{} center:{} right:{}".format(collision_free_prob[0], collision_free_prob[1], collision_free_prob[2]))
        print("Linear Velocity: {} Angular Velocity: {}".format(action.v_t,action.w))

        # Do more stuff.
        return action