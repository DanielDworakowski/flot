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
        loadpath = '/disk1/model/rl_multiconv-400x400-newdata-normalized_model_best.pth.tar'
        if mode == 'train':
            loadpath = None
        super(Config, self).__init__(MultiTraj_conv.Resnet_MultiConv(nSteps), loadPath = loadpath)
        # super(Config, self).__init__(MultiTraj_FC.Resnet_Multifc(nSteps), loadPath = loadpath)
        #
        # The distance threshold for postive / negative.
        self.distThreshold = 0.7
        #
        # How far to shift the image.
        self.hyperparam.shiftBounds = (110, 0)
        self.hyperparam.nSteps = nSteps
        self.hyperparam.numEpochs = 32
        self.hyperparam.cropShape = (400, 400)
        self.epochSaveInterval = 1
        self.modelSavePath = '/disk1/model/'
        #
        # Transforms.
        mean = [ 0.31102816, 0.30806204, 0.290305]
        std = [0.19752153, 0.19664317, 0.20129602]
        self.denormalize = DataUtil.UnNormalize(mean, std)
        self.normalize = DataUtil.Normalize(mean, std)
        self.transforms = transforms.Compose([
            Perterbations.RandomShift(self.hyperparam.cropShape, self.hyperparam.shiftBounds, self.hyperparam.nSteps, mode),
            Perterbations.RandomHorizontalFlip(0.5, mode),
            DataUtil.Rescale(self.hyperparam.image_shape),
            Perterbations.ColourJitter(0.05, 0.05, 0.05, 0.05, mode), # The effects of this must be tuned.
            DataUtil.ToTensor(),
            self.normalize,
        ])
        self.experimentName = 'rl_multiconv-400x400-newdata-normalized'
        self.dataValList = [
            '/disk1/rldata/20180304_042341',
            '/disk1/rldata/20180306_012910',
        ]
        self.dataTrainList = [
            '/disk1/rldata/20180310_215333',
            '/disk1/rldata/20180313_191316',
            '/disk1/rldata/20180314_001703',
            '/disk1/rldata/20180314_155329',
            '/disk1/rldata/20180314_172112',
            '/disk1/rldata/20180315_020652',
        ]
