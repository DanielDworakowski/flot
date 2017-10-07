from enum import Enum
import os
import time
from pathlib import Path
#
# Describing the supported environments.
class EnvironmentTypes(Enum):
    AirSim = 0
    Blimp = 1 # Unimplemented.
    Drone = 2 # Unimplemented.
#
# Default configuration that is overriden by subsequent configurations.
class DefaultConfig():
    #
    # The type of environment.
    envType = EnvironmentTypes.AirSim
    #
    # The agent that will be used.
    agentConstructor = None
    #
    # The file path to save to.
    savePath = './'
    #
    # The folder to save in.
    saveFolder = None
    #
    # The agent file name.
    agentType = 'testAgent'
    #
    # Save training data.
    serialize = True
    #
    # Retrieve the actual constructor.
    def getAgentConstructor(self):
        ag = __import__(self.agentType)
        self.agentConstructor = ag.Agent
    #
    # Transform relative to absolute paths.
    @staticmethod
    def getAbsPath(path):
        return os.path.abspath(path)
    #
    # Get the full save path.
    def getFullSavePath(self):
        if self.saveFolder == None:
            self.saveFolder = '/%s_%s_%s/'%(self.agentType, self.envType, time.strftime('%d-%m-%Y-%H-%M-%S'))
            self.savePath = self.savePath + self.saveFolder
        path = Path(self.savePath)
        path.mkdir(parents=True, exist_ok=True)
        return self.getAbsPath(self.savePath)
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
