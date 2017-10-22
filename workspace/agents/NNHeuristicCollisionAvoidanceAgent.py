from debug import *
import AgentBase as base
import Observations as obv
import os
import numpy as np
import torch
from torchvision import transforms
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
        # Load the model.
        if self.conf.modelLoadPath != None and os.path.isfile(self.conf.modelLoadPath):
            checkpoint = torch.load(self.conf.modelLoadPath)
            self.model = checkpoint['model']
            self.model.load_state_dict(checkpoint['state_dict'])
            printColour('Loaded model from path: %s'%self.conf.modelLoadPath, colours.OKBLUE)
        else:
            printError('Could not load model from path: %s', self.conf.modelLoadPath)
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
    # Image crop
    def cropImage(self, npimg):
        import pdb
        pdb.set_trace()
        return npimgs
    #
    # Reference to an observation
    def getActionImpl(self):
        obs = self.obs
        npimg = obs['img'].decompressPNG()[:,:,0:3]
        self.cropImage(npimg)
        img = Variable(self.toTensor(npimg).unsqueeze_(0).cuda())
        out = self.model(img)
        _, predicted = torch.max(out.data, 1)
        #
        # Do more stuff.
        return np.array([0,0,0])
