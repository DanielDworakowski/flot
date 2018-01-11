from enum import Enum
from torchvision import transforms, models
import pathlib
import torch
import torch.nn as nn
from nn.util import DataUtil
from nn.util import Perterbations
import os
from debug import *
from models import GenericModel
#
# The hyper parameters.
class HyperParam():
    #
    # Image shape
    image_shape = (224, 224, 3)
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
    # Scheduler.
    scheduler = None
    #
    # Network modification fn.
    networkModification = None
    ############################################################################
    ############################################################################
    ############################################################################
    #
    # How far to shift the image.
    shiftBounds = int(224/3)
    #
    # The number of shift bins.
    nSteps = (0, 0)
    def __init__(self, model):
        #
        # The model being used.
        self.model = model
        #
        # Optimizer.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#
# Default configuration that is overriden by subsequent configurations.
class DefaultConfig():
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
    networkModification = None
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
    # Initialization that may be different across configurations.
    ###########################################################################
    def __init__(self, model = GenericModel.GenericModel(models.resnet18(pretrained=True))):
        #
        # The hyper parameters.
        self.hyperparam = HyperParam(model)
        #
        # Create paths for saving models.
        pathlib.Path(self.modelSavePath).mkdir(parents=True, exist_ok=True)
        #
        # Check if cuda is available.
        # if not torch.cuda.is_available():
            # printError('CUDA is not available!')
        self.usegpu = (torch.cuda.is_available() and self.usegpu)
        if self.usegpu:
            self.hyperparam.model.cuda()
        #
        # Transforms.
        self.transforms = transforms.Compose([
            Perterbations.CenterCrop(self.hyperparam.image_shape),
            DataUtil.ToTensor(),
        ])

    def loadModel(self):
        ''' Load model from a specified directory.
        '''
        if self.modelLoadPath != None and os.path.isfile(self.modelLoadPath):
            checkpoint = torch.load(self.modelLoadPath)
            if type(checkpoint['model']) == type(self.hyperparam.model):
                self.hyperparam.model = checkpoint['model']
                self.hyperparam.model.load_state_dict(checkpoint['state_dict'])
                self.hyperparam.optimizer.load_state_dict(checkpoint['optimizer'])
                printColour('Loaded model from path: %s'%self.modelLoadPath, colours.OKBLUE)
            else:
                printError('Loaded model from path: %s is of type: (%s) while the specified model is of type: (%s)'%(self.modelLoadPath, type(checkpoint['model']), type(self.hyperparam.model)))
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
        self.dataTrainList = ['/home/rae/flot/workspace/data/test_dataset/']
