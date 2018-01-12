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
        nSteps = (5, 0)
        loadpath = '/disk1/model/10-01-2018-23-49-56_epoch_27.pth.tar'
        # loadpath = None
        super(Config, self).__init__(MultiTraj_FC.Resnet_Multifc(nSteps), loadPath = loadpath)
        self.hyperparam.nSteps = nSteps
        self.hyperparam.numEpochs = 32
        self.epochSaveInterval = 1
        self.modelSavePath = '/disk1/model/'
        # self.dataTrainList = ['/home/rae/flot/workspace/data/test_dataset/']
        # self.dataTrainList = [
        #     '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-00-38/',
        # ]
        #
        # Transforms.
        self.transforms = transforms.Compose([
            Perterbations.RandomShift(self.hyperparam.image_shape, self.hyperparam.shiftBounds, self.hyperparam.nSteps),
            DataUtil.ToTensor(),
        ])
        self.dataTrainList = [

# '/home/ddworakowski/flot/workspace/rnd0/daggerAgent_EnvironmentTypes.AirSim_03-12-2017-15-13-56/'

# '/home/ddworakowski/flot/workspace/rnd1/daggerAgent_EnvironmentTypes.AirSim_03-12-2017-15-07-25'

            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-00-38/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-41-26/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-01-50-26/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-01-30-02/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-21-02/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-10-50/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-03-01-50/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-01-19-50/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-51-38/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-03-12-02/',
            '/disk1/data/20171202-003453/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-00-34-59/',
            '/disk1/data/20171202-003453/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-00-55-23/',
            '/disk1/data/20171202-003453/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-01-05-35/',
            '/disk1/data/20171202-003453/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-00-45-11/',
            #
            # # Dagger 1.
            '/disk1/data/dagger1/20171202-115839/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-11-58-54/',
            '/disk1/data/dagger1/20171202-115410/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-11-54-25/',
            '/disk1/data/dagger1/20171202-113449/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-11-35-04/',
            '/disk1/data/dagger1/20171202-133243/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-13-32-59/',
            # #
            # # Dagger 2.
            # '/disk1/data/dagger2/20171202-140740/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-07-55/',
            # '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-14-50/',
            # '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-35-24/',
            # '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-25-06/',
            # '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-55-58/',
            # '/disk1/data/dagger2/20171202-135535/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-13-55-50/',
            # '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-45-41/',
        ]
        self.dataValList = [
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-31-14/',
            '/disk1/data/20171202-003453/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-01-15-47/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-03-22-14/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-03-32-26/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-03-42-38/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-01-40-14/',
        ]
