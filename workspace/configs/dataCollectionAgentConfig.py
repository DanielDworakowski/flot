import DefaultConfig
#
# Class to use the default configuration.
class Config(DefaultConfig.DefaultConfig):
    #
    # Where to load the model from.
    modelLoadPath = '/home/rae/flot/workspace/data/model/model_best.pth.tar'
    # modelLoadPath = '/home/user/Downloads/model/model_best.pth.tar'
    #
    # Image shape
    image_shape = (224, 224, 3)
    #
    # Save training data.
    serialize = True
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
