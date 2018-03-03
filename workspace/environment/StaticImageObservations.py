from debug import *
import Observations as observations
import cv2
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
        png_images = glob.glob('/home/jiwon/data/500test/*.png')
        png_images.sort()
        for filename in png_images:
            im=cv2.imread(filename) # RGB change
            self.images.append(im.copy())
        self.image_queue = cycle(self.images)


    def observeImpl(self, obs):
        obs.serializable['img'].uint8Img = next(self.image_queue)
        time.sleep(1)
        
        return True
