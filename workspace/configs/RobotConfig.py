from DefaultConfig import DefaultConfig, EnvironmentTypes

class Config(DefaultConfig):

    # Where to load the model from.
    # modelLoadPath = '/home/jiwon/data/model/model_best.pth.tar'
    # modelLoadPath = '/disk1/model/model_best.pth.tar'
    modelLoadPath = '/disk1/model/rl_normalized-rnet34_model_best.pth.tar'
    # modelLoadPath = '/disk1/model/09-03-2018-20-35-11_epoch_31_0.96.pth.tar'
    # modelLoadPath = '/home/kelvin/model_best.pth.tar'

    # Image shape
    image_shape = (224, 224, 3)

# Save training data.
    serialize = False

    # Initialize with DefaultConfig
    def __init__(self):
        super(Config, self).__init__()
        self.envType = EnvironmentTypes.Blimp
        self.agentType = 'RobotAgent'
