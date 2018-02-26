from debug import *
import Observations as observations
from PIL import Image
from itertools import cycle
import glob
import time
import os

# Recorder, run in seperate process.
class StaticImageObserver(observations.Observer):
    #
    # Constructor.
    def __init__(self, obsDir, serialize):
        observations.Observer.__init__(self, obsDir, serialize)
        #
        # Member variables.
        self.images = []
        for filename in glob.glob('/home/rae/data/20180223_220314/*.png'):
            im=Image.open(filename)
            self.images.append(im.copy())
            im.close()
        self.image_queue = cycle(self.images)


    def observeImpl(self, obs):
        time.sleep(1)
        obs.serializable['img'].pngImgs = next(self.image_queue)
        
        return True
