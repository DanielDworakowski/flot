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
        self.hyperparam.numEpochs = 32
        self.epochSaveInterval = 1

        self.modelSavePath = '/disk1/model/'
        # self.modelLoadPath = '/disk1/model/model_best.pth.tar'
        super(Config, self).loadModel()
        self.dataTrainList = [
'/disk1/data/20171119-022806/dumbAgent_EnvironmentTypes.AirSim_19-11-2017-02-28-16/',
'/disk1/data/20171119-033958/dumbAgent_EnvironmentTypes.AirSim_19-11-2017-03-50-25/',
'/disk1/data/20171119-033958/dumbAgent_EnvironmentTypes.AirSim_19-11-2017-03-40-08/',

        ]
        self.dataValList = [
'/disk1/data/20171119-031809/dumbAgent_EnvironmentTypes.AirSim_19-11-2017-03-18-20/',

        ]
