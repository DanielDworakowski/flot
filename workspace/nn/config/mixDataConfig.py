import torch
import os
from debug import *
import torch.nn as nn
from models import GenericModel
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
        nSteps = (0, 0)
        loadpath = '/disk1/model/rl_normalized-randomshift-lesscolour_model_best.pth.tar'
        # loadpath = '/disk1/model/rl_newDataFixedLabel_model_best.pth.tar'
        if mode == 'train':
            loadpath = None
        super(Config, self).__init__(model = GenericModel.GenericModel(models.resnet18(pretrained=True)), loadPath = loadpath)
        #
        # The distance threshold for postive / negative.
        self.distThreshold = 0.7
        #
        # How far to shift the image.
        self.hyperparam.shiftBounds = (30, 0)
        self.hyperparam.nSteps = nSteps
        self.hyperparam.numEpochs = 32
        self.hyperparam.cropShape = (448, 448)
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
            # Perterbations.CenterCrop(self.hyperparam.cropShape),
            DataUtil.Rescale(self.hyperparam.image_shape),
            # Perterbations.CenterCrop(self.hyperparam.image_shape),
            Perterbations.RandomHorizontalFlip(0.5, mode),
            Perterbations.ColourJitter(0.05, 0.05, 0.05, 0.05, mode), # The effects of this must be tuned.
            DataUtil.ToTensor(),
            self.normalize,
        ])
        self.experimentName = 'mixed-fixedData'
        # self.dataValList = ['/disk1/rldata/20180306_012910',]
        self.dataValList = [
            '/disk1/rldata/20180304_042341',
            '/disk1/rldata/20180306_012910',

            # SIM

            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-31-14/',
            '/disk1/data/20171202-003453/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-01-15-47/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-03-22-14/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-03-32-26/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-03-42-38/',
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-01-40-14/',
        ]
        self.dataTrainList = [
            '/disk1/rldata/20180310_215333',
            '/disk1/rldata/20180313_191316',
            '/disk1/rldata/20180314_001703',
            '/disk1/rldata/20180314_155329',
            '/disk1/rldata/20180314_172112',
            '/disk1/rldata/20180315_020652',
            '/disk1/rldata/20180317_030020',

            #
            # SIM
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
            '/disk1/data/dagger2/20171202-140740/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-07-55/',
            '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-14-50/',
            '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-35-24/',
            '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-25-06/',
            '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-55-58/',
            '/disk1/data/dagger2/20171202-135535/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-13-55-50/',
            '/disk1/data/dagger2/20171202-141434/dataCollectionAgent_EnvironmentTypes.AirSim_02-12-2017-14-45-41/',

        ]
