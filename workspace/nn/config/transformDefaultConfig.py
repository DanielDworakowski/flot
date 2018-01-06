from enum import Enum
from torchvision import transforms, models
import pathlib
import torch
import torch.nn as nn
from nn.util import DataUtil
from nn.util import Perterbations
import os
from debug import *
#
# The hyper parameters.
class HyperParam():
    #
    # The model being used.
    model = models.resnet18(pretrained=True)
    #
    # Image shape
    image_shape = (224, 224, 3)
    #
    # How far to shift the image.
    shiftBounds = int(224/3)
    #
    # The number of shift bins.
    nSteps = (10, 0)
    #
    # Number of images in a batch.
    batchSize = 32
    #
    # How many epochs to train for.
    numEpochs = 10
    #
    # Criteria.
    criteria = nn.CrossEntropyLoss()
    #
    # Optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #
    # Scheduler.
    scheduler = None
    #
    # The training signals.
    trainSignals = ['trajectoryIndicator']
    #
    # Network modification fn.
    networkModification = None
#
# Resize the network.
def resizeFC(net, param):
    numFeat = net.fc.in_features
    net.fc = nn.Linear(numFeat, len(param.trainSignals) + 1) # need for positive and negative class.
#
# Default configuration that is overriden by subsequent configurations.
class DefaultConfig():
    #
    # The hyper parameters.
    hyperparam = HyperParam()
    #
    # The default data path.
    dataTrainList = [
    ]
    #
    # The default validation set.
    dataValList = [
        # '/disk1/val/data1'
    ]
    #
    # The csv file name.
    csvFileName = 'labels.csv'
    #
    # The image type name.
    imgName = 'front_camera'
    #
    # Transforms.
    transforms = transforms.Compose([
        Perterbations.RandomShift(hyperparam.image_shape, hyperparam.shiftBounds, hyperparam.nSteps),
        DataUtil.ToTensor(),
    ])
    #
    # Transform relative to absolute paths.
    @staticmethod
    def getAbsPath(path):
        return os.path.abspath(path)
    #
    # Doesnt usually need to be changed.
    usegpu = True
    #
    # Save tensorboard data.
    useTensorBoard = False
    #
    # Number of workers for loading data.
    numWorkers = 8
    #
    # Resize the network as needed.
    networkModification = resizeFC
    #
    # Save every x epochs.
    epochSaveInterval = 1
    #
    # Model save path.
    modelSavePath = ''
    #
    # Load a model.
    modelLoadPath = None
    ###########################################################################
    # Functions to run config.
    ###########################################################################
    #
    # Create paths for saving models.
    pathlib.Path(modelSavePath).mkdir(parents=True, exist_ok=True)
    #
    # Run the resize.
    if networkModification != None:
        networkModification(hyperparam.model, hyperparam)
    #
    # Check if cuda is available.
    # if not torch.cuda.is_available():
        # printError('CUDA is not available!')
    usegpu = (torch.cuda.is_available() and usegpu)
    if usegpu:
        hyperparam.model.cuda()
    def loadModel(self):
        ''' Load model from a specified directory.
        '''
        if self.modelLoadPath != None and os.path.isfile(self.modelLoadPath):
            checkpoint = torch.load(self.modelLoadPath)
            self.hyperparam.model = checkpoint['model']
            self.hyperparam.model.load_state_dict(checkpoint['state_dict'])
            self.hyperparam.optimizer.load_state_dict(checkpoint['optimizer'])
            printColour('Loaded model from path: %s'%self.modelLoadPath, colours.OKBLUE)
        elif self.modelLoadPath != None:
            printError('Unable to load specified model: %s'%(self.modelLoadPath))

#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
        self.modelSavePath = '/disk1/model/'
        # self.dataTrainList = ['/home/rae/flot/workspace/data/test_dataset/']
        self.dataTrainList = [
            '/disk1/data/20171202-011944/dumbAgent_EnvironmentTypes.AirSim_02-12-2017-02-00-38/',
        ]
