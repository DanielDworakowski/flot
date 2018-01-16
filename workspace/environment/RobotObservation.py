# Removes conflict regarding CV2 if ROS Kinetic has been sourced
import os

# from multiprocessing import Process, Queue
import Observations as observations

# Libraries for video stream client
import sys, time, threading
import numpy as np
from subprocess import Popen, PIPE
from shlex import split

# Removes conflict regarding CV2 if ROS Kinetic has been sourced
del os.environ['PYTHONPATH']
import cv2

# Recorder, run in seperate process.
class RobotObserver(observations.Observer):
    # Constructor.
    def __init__(self, obsDir):
        observations.Observer.__init__(self, obsDir)

        # Member variables.
        self.stream = VideoStreamClient()
        self.collInfo = None
        self.imgs = None
        self.stream.start()

    def observeImpl(self, obs):
        obs.valid = True
        obs.serializable['timestamp'].val = time.time()
        obs.serializable['img'].uint8Img = self.stream.frame


class VideoStreamClient(threading.Thread):

    VERBOSE = None

    nc_cmd = 'nc -l '
    port = None
    ffmpeg_cmd = [ 'ffmpeg',
            '-i', 'pipe:0',           # Use stdin pipe for input
            '-pix_fmt', 'bgr24',      # opencv requires bgr24 pixel format.
            '-vcodec', 'rawvideo',
            '-an','-sn',              # we want to disable audio processing (there is no audio)
            '-f', 'image2pipe', '-']

    # Pipe buffer size calculation for image size
    width = None
    height = None
    depth = None
    num = None
    bufsize = None

    # Pipes, processes, thread
    nc_pipe = None
    pipe = None
    thread = None

    frame = None

    # Class constructor
    def __init__(self, port='2224', width=640, height=480, depth=3, num=2, VERBOSE=False):

        threading.Thread.__init__(self)

        self.port = port
        self.width = width
        self.height = height
        self.depth = depth
        self.num = num
        self.bufsize = width*height*depth*num

        self.VERBOSE = VERBOSE

    def getNCCommand(self):
        return split(self.nc_cmd + self.port)

    # Action when thread is started
    def run(self):
        self.nc_pipe = Popen(self.getNCCommand(), stdout=PIPE)
        self.pipe = Popen(self.ffmpeg_cmd,
                        stdin=self.nc_pipe.stdout,
                        stdout=PIPE,
                        bufsize=self.bufsize)

        if self.VERBOSE:
            print('Listening for stream...')

        while True:
            # Capture frame bytes from pipe
            raw_image = self.pipe.stdout.read(self.width*self.height*self.depth)

            # Transform bytes to numpy array
            image =  np.fromstring(raw_image, dtype='uint8')
            image = image.reshape((self.height, self.width, self.depth))

            if image is not None:
                # Convert image from BGR to RGB
                self.frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.VERBOSE:
                    cv2.imshow('Video', image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            self.pipe.stdout.flush()

        cv2.destroyAllWindows()


## UNCOMMENT BELOW FOR DEBUGGING
# if __name__ == '__main__':
#
#     # Removes conflict regarding CV2 if ROS Kinetic has been sourced
#     del os.environ['PYTHONPATH']
#     import cv2
#
#     client = VideoStreamClient()
#     client.start()
#
#     while True:
#         if client.frame is not None:
#             cv2.imshow('Video', client.frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
