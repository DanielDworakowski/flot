import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
import itertools

class A2CValueNetwork(torch.nn.Module):
    def __init__(self):
        super(A2CValueNetwork, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
  
    def forward(self, x):
        return self.model(x)
  