import Observations as obs
from enum import Enum
from debug import *
from DefaultConfig import EnvironmentTypes
#
# Options class
class EnvironmentOptions():
    #
    # Configuration for each of the types.
    def getAirSimConfig():
        import AirSimObservations as obs
        observer = obs.AirSimObserver
        actionClient = None
        printError('Air sim config')
        # actionClient = act.Action()
        return [observer, actionClient]
    #
    # Blimp configuration options.
    def getBlimpConfig():
        raise NotImplementedError
    #
    # Drone configuration options.
    def getDroneConfig():
        raise NotImplementedError
    #
    # Return the configuration options for each environment.
    options = {
        EnvironmentTypes.AirSim: getAirSimConfig,
        EnvironmentTypes.Blimp: getBlimpConfig,
        EnvironmentTypes.Drone: getDroneConfig,
    }
#
# The environment dictates how actions are performed and the observations recieved.
class Environment():
    #
    # Constructor.
    def __init__(self, envType, saveDirectory, serialize):
        #
        # Configuration.
        self.serialize = serialize
        #
        # Get the correct class types.
        try:
            self.observer = None
            self.actionEngine = None
            obs, act = EnvironmentOptions.options[envType]()
            #
            # Check init.
            if obs == None or act == None:
                printError('Could not initialize environment.')
            self.observer = obs(saveDirectory)
            # self.actionEngine = act()
        except KeyError:
            printError('Passed type does not have a configuration.')
            raise KeyError
    #
    # Open observation file.
    def __enter__(self):
        self.observer.__enter__()
        return self
    #
    # Close the observation file.
    def __exit__(self, type, value, traceback):
        self.observer.__exit__(type, value, traceback)
        return False
    #
    # Do action.
    def runAction(self, action):
        printError('run action here.')
        # return self.actionEngine.act(action)
    #
    # Get observation.
    def observe(self):
        self.observer.observe()
        if self.serialize:
            self.observer.serialize()
        return self.observer.obs
