# Libraries for video stream client
import sys, time, threading
import numpy as np
from subprocess import Popen, PIPE
from shlex import split
import cv2

class VideoStreamClient(threading.Thread):

    VERBOSE = None
    BGR2RGB = None

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
    def __init__(self, port='2222', width=640, height=480, depth=3, num=2, VERBOSE=False, BGR2RGB=False):

        threading.Thread.__init__(self)

        self.port = port
        self.width = width
        self.height = height
        self.depth = depth
        self.num = num
        self.bufsize = width*height*depth*num

        self.VERBOSE = VERBOSE
        self.BGR2RGB = BGR2RGB

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
            print('Listening for video stream...')

        while True:
            # Capture frame bytes from pipe
            raw_image = self.pipe.stdout.read(self.width*self.height*self.depth)

            # Transform bytes to numpy array
            image =  np.fromstring(raw_image, dtype='uint8')
            image = image.reshape((self.height, self.width, self.depth))

            if image is not None:
                # Convert image from BGR to RGB
                if self.BGR2RGB:
                    self.frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    self.frame = image

                if self.VERBOSE:
                    cv2.imshow('Video', image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            self.pipe.stdout.flush()

        cv2.destroyAllWindows()
