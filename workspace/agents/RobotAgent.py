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
        # Check if cuda is available.
        self.usegpu = torch.cuda.is_available() and self.conf.usegpu
        #
        # Load the model.
        if self.conf.modelLoadPath != None and os.path.isfile(self.conf.modelLoadPath):
            if self.usegpu:
                checkpoint = torch.load(self.conf.modelLoadPath)
            else:
                print('Running the agent on CPU!')
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
        self.t = transforms.Compose([transforms.ToTensor()])
        if any(isinstance(tf, DataUtil.Rescale) for tf in self.t.transforms):
            self.model_input_img_shape = (self.nnconf.cropshape[0],self.nnconf.cropshape[1],3)
            self.t = transforms.Compose([
                transforms.Rescale(conf.hyperparam.image_shape),
                transforms.ToTensor(),
            ])
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
        self.back_min_prob = 0.20
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
        npimg = obs['img'].uint8Img

        # If image is available
        if npimg is not None:
            cropped_imgs = self.cropImageToThree(npimg)
            collision_free_prob = []
            softmax = torch.nn.Softmax()
            probs = None

            # Runs classification over each cropped image
            for idx, cropped_img in enumerate(cropped_imgs):
                img = self.t(cropped_img)
                if self.usegpu:
                    img = Variable(img.unsqueeze_(0).cuda(async=True))
                else:
                    img = Variable(img.unsqueeze_(0))

                classActivation = self.model(img)
                collision_free_pred = classActivation.data

                if idx == 1:
                    probs = self.model.getClassifications(classActivation, softmax)

                collision_free_prob.append(softmax(classActivation)[0,1].data.cpu().numpy())

            # collision free probability
            left_prob, center_prob, right_prob = collision_free_prob
            left_prob, center_prob, right_prob = [left_prob[0], center_prob[0], right_prob[0]]
            action_array = np.zeros(self.action_array_dim)
            if center_prob > self.straight_min_prob:
                action_array[int(self.action_array_dim/2)] = 1
                action = Action(action_array)

            elif left_prob<self.stop_min_prob and center_prob<self.stop_min_prob and right_prob<self.stop_min_prob:
                action = Action(action_array)

            elif center_prob < self.back_min_prob:
                action = Action(v_t=-1*self.max_v_t,w=0)
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
            print("Linear Velocity: {:.2f} Angular Velocity: {:.2f}".format(action.v_t,action.w))

            # Place the activations for visualization.
            action.meta['activations'] = probs.cpu().numpy()[0]

        # Take no action when no image is available
        else:
            action = Action(v_t=0,w=0)

        # Do more stuff.
        return action
