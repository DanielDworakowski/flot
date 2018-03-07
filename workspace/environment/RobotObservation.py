import Observations as observations
from util import RobotUtil

# Libraries for video stream client
import sys, time, threading
import numpy as np
from subprocess import Popen, PIPE
from shlex import split
import cv2

# Recorder, run in seperate process.
class RobotObserver(observations.Observer):
    # Constructor.
    def __init__(self, obsDir, serialize):
        observations.Observer.__init__(self, obsDir, serialize)

        # Member variables.
        self.stream = RobotUtil.VideoStreamClient(BGR2RGB=True)
        self.stream.daemon=True
        self.collInfo = None
        self.imgs = None
        self.stream.start()

    def observeImpl(self, obs):
        obs.valid = True
        obs.serializable['timestamp'].val = time.time()
        obs.serializable['img'].uint8Img = self.stream.getFrame()

    def __del__(self):
        self.stream.terminate()
        self.stream.join()