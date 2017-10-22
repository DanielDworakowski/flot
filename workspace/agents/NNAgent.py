from debug import *
from Actions import Action
import Observations as obv
import AgentBase
import os
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
#
# Neural network agent class.
class Agent(AgentBase.AgentBase):
    #
    # Constructor.
    def __init__(self, conf):
        super(Agent, self).__init__(conf)
        self.conf = conf
        self.toTensor = transforms.ToTensor()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet values
            # DataUtil.Normalize([0.08086318, 0.09237641,  0.12678191], [ 0.08651822,  0.09291226,  0.10738404])
        ])
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
    #
    # Reference to an observation
    def getActionImpl(self):
        obs = self.obs
        npimg = obs['img'].decompressPNG()[:,:,0:3]
        #
        # NORMALIZE IMAGE.
        # img = Variable(self.toTensor(npimg).unsqueeze_(0).cuda())
        img = Variable(self.transforms(npimg).unsqueeze_(0).cuda())
        out = self.model(img)
        _, predicted = torch.max(out.data, 1)
        #
        # Since we are doing a 1,0 encoding take the max and push it through.
        move = predicted.cpu().numpy().tolist()
        print(move)
        act = Action(v_t = move[0]*1.0, w = 0.0)
        return act
