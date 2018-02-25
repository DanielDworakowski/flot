from debug import *
import Observations as observations
from PIL import Image
from itertools import cycle
import glob

# Recorder, run in seperate process.
class StaticImageObserver(observations.Observer):
    #
    # Constructor.
    def __init__(self, obsDir, serialize):
        observations.Observer.__init__(self, obsDir, serialize)
        #
        # Member variables.
        self.images = []
        for filename in glob.glob('obsDir/*.png'):
            im=Image.open(filename)
            images.append(im)
        self.image_queue = cycle(self.images)


    def observeImpl(self, obs):
        obs.serializable['img'].pngImgs = next(self.image_queue)
        
        return True
