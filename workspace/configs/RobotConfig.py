from DefaultConfig import DefaultConfig, EnvironmentTypes

class Config(DefaultConfig):

    # Where to load the model from.
    modelLoadPath = '/home/jiwon/data/model/model_best.pth.tar'

    # Image shape
    image_shape = (224, 224, 3)

    # Save training data.
    serialize = False

    # Initialize with DefaultConfig
    def __init__(self):
        super(Config, self).__init__()
        self.envType = EnvironmentTypes.Blimp
        self.agentType = 'RobotAgent'
