import DefaultConfig
#
# Class to use the default configuration.
class Config(DefaultConfig.DefaultConfig):
    #
    # Where to load the model from.
    modelLoadPath = '/disk1/model/14-10-2017-11-28-45_epoch_0.pth.tar'
    #
    # Initialize.
    def __init__(self):
        super(Config, self).__init__()
