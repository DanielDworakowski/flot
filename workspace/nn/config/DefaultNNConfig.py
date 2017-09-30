from enum import Enum
import os
#
# Default configuration that is overriden by subsequent configurations.
class DefaultConfig():
    #
    # The default data path.
    dataPath = {
        '/disk1' : [
            'data1'
        ]
    }
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
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
