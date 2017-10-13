from enum import Enum
from torchvision import transforms, models
import torch
import torch.nn as nn
import DataUtil
import os
#
# The hyper parameters.
class HyperParam():
    #
    # The model being used.
    model = models.resnet18(pretrained=True)
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
    net.fc = nn.Linear(numFeat, len(param.trainSignals))
#
# Default configuration that is overriden by subsequent configurations.
class DefaultConfig():
    #
    # The hyper parameters.
    hyperparam = HyperParam()
    #
    # The default data path.
    dataTrainList = [
        '/disk1/data/testAgent_EnvironmentTypes.AirSim_07-10-2017-18-29-03'
    ]
    #
    # The default validation set.
    dataValList = [
        # '/disk1/val/data1'
    ]
    #
    # The csv file name.
    csvFileName = 'observations.csv'
    #
    # The image type name.
    imgName = 'front_camera'
    #
    # Transforms.
    transforms = transforms.Compose([
        DataUtil.ToTensor(),
        DataUtil.Normalize([0.0, 0.0, 0.0], [1, 1, 1])
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
    ###########################################################################
    # Functions to run config.
    ###########################################################################
    #
    # Run the resize.
    if networkModification != None:
        networkModification(hyperparam.model, hyperparam)
    #
    # Check if cuda is available.
    if not torch.cuda.is_available():
        printError('CUDA is not available!')
    usegpu = (torch.cuda.is_available() and usegpu)
    if usegpu:
        hyperparam.model.cuda()

#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
