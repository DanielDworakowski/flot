import DefaultConfig
#
# Class to use the default configuration.
class Config(DefaultConfig.DefaultConfig):
    #
    # Where to load the model from.
    modelLoadPath = '/disk1/model/21-10-2017-17-01-31_epoch_9.pth.tar'
    #
    # Save training data.
    serialize = False
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
