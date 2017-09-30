from PythonClient import *
from debug import *
from multiprocessing import Process, Queue
import Observations as observations
import cv2
#
# Recorder, run in seperate process.
class AirSimObserver(observations.Observer):
    #
    # Constructor.
    def __init__(self, obsDir):
        observations.Observer.__init__(self, obsDir)
        #
        # Member variables.
        self.client = MultirotorClient()
        self.collInfo = None
        self.imgs = None
    #
    # Get a single image from the simulator.
    def getSingleImage(self):
        stat = True
        self.imgs = self.client.simGetImages([
            ImageRequest(0, AirSimImageType.Scene, pixels_as_float = False, compress = True)])
        if self.imgs == None or self.imgs == []:
            stat = False
            printError('Failed to retrieve an image.')
        return stat
    #
    # Get the CollisionInfo.
    def getCollisionInfo(self):
        self.collInfo = self.client.getCollisionInfo()
        return not (self.collInfo == None)
    #
    # Fill the values.
    @staticmethod
    def fillOrientation(strdict):
        q = Quaternionr()
        q.w_val = strdict[b'w_val']
        q.x_val = strdict[b'x_val']
        q.y_val = strdict[b'y_val']
        q.z_val = strdict[b'z_val']
        return AirSimClientBase.toEulerianAngle(q)
    #
    # Fill in the observations.
    def fillObservations(self, obs):
        obs.valid = True
        img = self.imgs[0]
        cameraPos = img.camera_position
        cameraOrient = self.fillOrientation(img.camera_orientation)
        obs.serializable['timestamp_ns'].val = img.time_stamp
        obs.serializable['img'].pngImgs = self.imgs
        obs.serializable['cameraPosition'].x = cameraPos[b'x_val']
        obs.serializable['cameraPosition'].y = cameraPos[b'y_val']
        obs.serializable['cameraPosition'].z = cameraPos[b'z_val']
        obs.serializable['cameraRotation'].pitch = cameraOrient[0]
        obs.serializable['cameraRotation'].roll = cameraOrient[1]
        obs.serializable['cameraRotation'].yaw = cameraOrient[2]
        obs.serializable['hasCollided'].val = self.collInfo.has_collided
    #
    # Fills in the observations data structure.
    def observeImpl(self, obs):
        # obs = observations.Observations()
        obs.valid = False
        #
        # Obtain all sensor data.
        stat = True
        stat &= self.getSingleImage()
        stat &= self.getCollisionInfo()
        if not stat:
            printError('Failed to fill in observations')
            return False
        #
        # Ensure that the timestamp difference is reasonable.
        diff = self.collInfo.time_stamp - self.imgs[0].time_stamp
        # if np.abs(diff) > 1e-4:
        #     printError('Timestamp difference between observation is too great : %s [s]'%(diff))
        #     return False
        #
        # Fill in all of the data.
        self.fillObservations(obs)
        return True
