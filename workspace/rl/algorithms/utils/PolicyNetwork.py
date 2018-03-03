import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
import itertools

class A2CPolicyNetwork(torch.nn.Module):
    def __init__(self, action_dim):
        super(A2CPolicyNetwork, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, action_dim)
  
    def forward(self, x):
        return self.model(x)
  