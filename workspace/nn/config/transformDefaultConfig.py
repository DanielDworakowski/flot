from enum import Enum
from torchvision import transforms, models
import pathlib
import torch
import torch.nn as nn
from nn.util import DataUtil
from nn.util import Perterbations
import os
from debug import *
from config.DefaultNNConfig import DefaultConfig
from models import MultiTraj_FC
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self):
        nSteps = (1, 1)
        super(Config, self).__init__(MultiTraj_FC.Resnet_Multifc(nSteps))
        self.hyperparam.nSteps = nSteps
        self.hyperparam.numEpochs = 32
        self.epochSaveInterval = 1
        self.modelSavePath = '/disk1/model/'
        # self.dataTrainList = ['/home/rae/flot/workspace/data/test_dataset/']
        self.dataTrainList = [
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-00-38/',
        ]
        #
        # Transforms.
        self.transforms = transforms.Compose([
            Perterbations.RandomShift(self.hyperparam.image_shape, self.hyperparam.shiftBounds, self.hyperparam.nSteps),
            DataUtil.ToTensor(),
        ])
