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
# '/disk1/data/20171119-022806/dumbAgent_EnvironmentTypes.AirSim_19-11-2017-02-28-16/',
# '/disk1/data/20171119-223445/dumbAgent_EnvironmentTypes.AirSim_19-11-2017-22-34-56/',
# '/disk1/data/20171119-224222/dumbAgent_EnvironmentTypes.AirSim_19-11-2017-22-42-32/',
#
'/disk1/data/20171125-004744/daggerAgent_EnvironmentTypes.AirSim_25-11-2017-00-47-58/',
#
# old
# '/disk1/oldData/20171119-000214/daggerAgent_EnvironmentTypes.AirSim_19-11-2017-00-02-29',
# '/disk1/oldData/20171118-174701/daggerAgent_EnvironmentTypes.AirSim_18-11-2017-17-47-16',
# '/disk1/oldData/20171118-124720/daggerAgent_EnvironmentTypes.AirSim_18-11-2017-12-47-32',

        ]
        self.dataValList = [
'/disk1/data/20171119-031809/dumbAgent_EnvironmentTypes.AirSim_19-11-2017-03-18-20/',
        ]
