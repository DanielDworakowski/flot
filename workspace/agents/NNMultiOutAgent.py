from debug import *
import AgentBase as base
import Observations as obv
import os
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from Actions import Action
from PIL import Image
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
        self.transform = transforms.Compose([
            transforms.CenterCrop(self.nnconf.cropShape),
            transforms.Resize(self.nnconf.hyperparam.image_shape),
            transforms.ToTensor()
        ])
    #
    # Reference to an observation
    def getActionImpl(self):
        obs = self.obs
        npimg = Image.fromarray(obs['img'].decompressPNG()[:,:,0:3])
        # npimg.show()
        # print(npimg.size)
        inputImg = self.transform(npimg)
        sm = torch.nn.Softmax()
        if self.usegpu:
            img = Variable(inputImg.unsqueeze_(0).cuda())
        else:
            img = Variable(inputImg.unsqueeze_(0))
        classActivation = self.model(img).data
        probs = self.model.getClassifications(Variable(classActivation), sm)
        # action = Action(probs.cpu().numpy())
        action = Action(np.zeros(3))
        action.meta['activations'] = probs.cpu().numpy()[0]

        return action
