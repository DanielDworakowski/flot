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
        import AirSimActions as act
        observer = obs.AirSimObserver
        actionClient = act.AirSimActionEngine
        return [observer, actionClient]
    #
    # Blimp configuration options.
    def getBlimpConfig():
        import RobotObservation as obs
        import sys
        if sys.version_info[0] < 3:
            import RobotActionsP27 as act
        else:
            import RobotActions as act
        observer = obs.RobotObserver
        actionClient = act.RobotActionEngine
        return [observer, actionClient]
    #
    # Drone configuration options.
    def getDroneConfig():
        raise NotImplementedError
    #
    # Static configuration options.
    def getStaticConfig():
        import StaticImageObservations as obs
        import StaticImageActions as act
        observer = obs.StaticImageObserver
        actionClient = act.StaticActionEngine
        return [observer, actionClient]
    #
    # Return the configuration options for each environment.
    options = {
        EnvironmentTypes.AirSim: getAirSimConfig,
        EnvironmentTypes.Blimp: getBlimpConfig,
        EnvironmentTypes.Drone: getDroneConfig,
        EnvironmentTypes.Static: getStaticConfig
    }
#
# The environment dictates how actions are performed and the observations recieved.
class Environment(object):
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
            self.observer = obs(saveDirectory, self.serialize)
            self.actionEngine = act()
        except KeyError:
            printError('Passed type does not have a configuration.')
            raise KeyError
    #
    # Open observation file.
    def __enter__(self):
        self.observer.__enter__()
        self.actionEngine.__enter__()
        return self
    #
    # Close the observation file.
    def __exit__(self, type, value, traceback):
        self.observer.__exit__(type, value, traceback)
        self.actionEngine.__exit__(type, value, traceback)
        return False

    # reset if requested
    def reset(self, action):
        #fix later
        pose = action.pose
        if pose==1: self.actionEngine.reset()
        elif pose: self.actionEngine.reset(pose)
    #
    # Do action.
    def runAction(self, action, obs):
        # self.reset(action)
        return self.actionEngine.executeAction(action, obs)
    #
    # Get observation.
    def observe(self):
        self.observer.observe()
        if self.serialize:
            self.observer.serialize()
        return self.observer.obs
