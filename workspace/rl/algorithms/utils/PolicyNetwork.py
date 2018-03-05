import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
from PIL import Image
import itertools

import pdb

class A2CPolicyNetwork(torch.nn.Module):
    def __init__(self, dtype, action_dim):
        super(A2CPolicyNetwork, self).__init__()
        self.dtype = dtype
        self.action_dim = action_dim
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.action_dim*2)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224), interpolation=Image.CUBIC), transforms.ToTensor()])
  
    def forward(self, x):
        return self.model(x)

    def compute(self, observation):
        observation = torch.autograd.Variable(self.transform(observation)).type(self.dtype.FloatTensor).unsqueeze(0)
        # plt.imshow(observation.data.cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')        
        model_out = self.forward(observation).squeeze()
        mean, std_dev = model_out[:self.action_dim].data, torch.exp(model_out[self.action_dim:].data)
        distribution = torch.distributions.Normal(mean, std_dev)
        return distribution.sample().cpu().numpy()
  
    def train(self):
        pass
