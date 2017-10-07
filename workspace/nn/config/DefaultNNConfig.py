from enum import Enum
from torchvision import transforms, models
import torch
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
    # Criteria.
    criterion = nn.CrossEntropyLoss()
    #
    # Optimizer.
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #
    # Scheduler.
    scheduler = None
#
# Default configuration that is overriden by subsequent configurations.
class DefaultConfig():
    #
    # The hyper parameters.
    hyperparam = HyperParam()
    #
    # The default data path.
    dataTrainList = [
        '/disk1/data/testAgent_EnvironmentTypes.AirSim_06-10-2017-20-51-32/'
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
    # Number of workers for loading data.
    numWorkers = 8
    #
    # Check if cuda is available.
    if not torch.cuda.is_available():
        printError('CUDA is not available!')
    useGpu = (torch.cuda.is_available() and conf.usegpu)

#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
