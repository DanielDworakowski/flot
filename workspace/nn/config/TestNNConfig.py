from enum import Enum
from torchvision import transforms, models
import pathlib
import torch
import torch.nn as nn
import DataUtil
import os
from DefaultNNConfig import DefaultConfig
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
        self.hyperparam.numEpochs = 10
        self.modelSavePath = '/home/user/workspace/data/'
        self.dataTrainList = ['/home/user/workspace/data/test_dataset']
