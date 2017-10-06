from enum import Enum
from torchvision import transforms
import DataUtil
import os
#
# The hyper parameters.
class HyperParam():
    #
    # Number of images in a batch.
    batchSize = 32
#
# Default configuration that is overriden by subsequent configurations.
class DefaultConfig():
    #
    # The hyper parameters.
    hyperparam = HyperParam()
    #
    # The default data path.
    dataTrainList = [
        '/disk1/data1'
    ]
    #
    # The default validation set.
    dataValList = [
        '/disk1/val/data1'
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
        transforms.Normalize([0.0, 0.0, 0.0], [1, 1, 1])
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
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
