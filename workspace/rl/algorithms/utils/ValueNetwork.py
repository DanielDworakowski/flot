import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
from PIL import Image
import itertools

import pdb

class A2CValueNetwork(torch.nn.Module):
    def __init__(self, dtype):
        super(A2CValueNetwork, self).__init__()
        self.dtype = dtype
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224), interpolation=Image.CUBIC), transforms.ToTensor()])
  
    def forward(self, x):
        return self.model(x)

    def compute(self, observations):
        pdb.set_trace()
        observations  = [self.transform(obs) for obs in observations]
        observations = torch.autograd.Variable(torch.stack(observations)).type(self.dtype.FloatTensor)      
        model_out = self.forward(observation)

        return self.forward(x)
  