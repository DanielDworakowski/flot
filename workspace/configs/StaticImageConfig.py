from DefaultConfig import DefaultConfig, EnvironmentTypes
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Where to load the model from.
    # modelLoadPath = '/home/user/workspace/data/model_best.pth.tar'
    # modelLoadPath = '/home/tommy/Downloads/model/model_best.pth.tar'
    modelLoadPath = '/home/rae/data/model/model_best.pth.tar'
    #
    # Image shape
    image_shape = (224, 224, 3)
    #
    # Save training data.
    serialize = False
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
        self.envType = EnvironmentTypes.Static
        self.agentType = 'StaticAgent'
