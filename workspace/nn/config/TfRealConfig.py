import torch
import os
from debug import *
import torch.nn as nn
from models import MultiTraj_FC
from models import MultiTraj_conv
from torchvision import transforms, models
from nn.util import DataUtil, Perterbations
from config.DefaultNNConfig import DefaultConfig
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self, mode = 'train'):
        nSteps = (2, 0)
        loadpath = '/disk1/model/rl_data-noscale_model_best.pth.tar'
        if mode == 'train':
            loadpath = None
        super(Config, self).__init__(MultiTraj_conv.Resnet_MultiConv(nSteps), loadPath = loadpath)
        #
        # The distance threshold for postive / negative.
        self.distThreshold = 0.7
        #
        # How far to shift the image.
        self.hyperparam.shiftBounds = (90, 0)
        self.hyperparam.nSteps = nSteps
        self.hyperparam.numEpochs = 32
        self.hyperparam.cropShape = (448, 448)
        self.epochSaveInterval = 1
        self.modelSavePath = '/disk1/model/'
        #
        # Transforms.
        self.transforms = transforms.Compose([
            Perterbations.RandomShift(self.hyperparam.cropShape, self.hyperparam.shiftBounds, self.hyperparam.nSteps, mode),
            Perterbations.RandomHorizontalFlip(0.5, mode),
            DataUtil.Rescale(self.hyperparam.image_shape),
            Perterbations.ColourJitter(0.25, 0.25, 0.25, 0.1, mode), # The effects of this must be tuned.
            DataUtil.ToTensor(),
        ])
        self.experimentName = 'rl_multiconv'
        self.dataValList = ['/disk1/rldata/20180223_220314',]
        self.dataTrainList = [
        '/disk1/rldata/20180306_012910',
        '/disk1/rldata/20180304_042341'
        ]