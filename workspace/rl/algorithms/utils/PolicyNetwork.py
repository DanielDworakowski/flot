import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
import itertools

class A2CPolicyNetwork(torch.nn.Module):
    def __init__(self, action_dim):
        super(A2CPolicyNetwork, self).__init__()
        self.action_dim = action_dim
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.action_dim*2)
  
    def forward(self, x):
        return self.model(x)

    def compute(self, observation):
        model_out = self.forward(observation)
        mean, std_dev = model_out[:self.action_dim], model_out[self.action_dim:]
        distribution = torch.distributions.Normal(mean, std_dev)
        return distribution.sample()
  
    def train(self)