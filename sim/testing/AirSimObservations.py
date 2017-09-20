from PythonClient import *
from debug import *
from multiprocessing import Process, Queue
import Observations as observations
import cv2
#
# Recorder, run in seperate process.
class AirSimObservations(observations.Observer):
    #
    # Constructor.
    def __init__(self):
        observations.Observer.__init__(self)
        #
        # Member variables.
        self.client = AirSimClient()
        self.collInfo = None
        self.imgs = None
    #
    # Serialize image data.
    @staticmethod
    def saveImages(imgs):
        for img in imgs:
            if img.pixels_as_float:
                print("Type %d, size %d" % (img.image_type, len(img.image_data_float)))
                AirSimClient.write_pfm(os.path.normpath('/home/ddworakowski/flot/sim/testing/py1.pfm'), AirSimClient.getPfmArray(img))
            else:
                print("Type %d, size %d" % (img.image_type, len(img.image_data_uint8)))
                AirSimClient.write_file(os.path.normpath('/home/ddworakowski/flot/sim/testing/py1.png'), img.image_data_uint8)
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
        return AirSimClient.toEulerianAngle(q)
    #
    # Fill in the observations.
    def fillObservations(self, obs):
        img = self.imgs[0]
        cameraPos = img.camera_position
        print(cameraPos)
        print(img.pixels_as_float)
        print(img.width)
        print(img.height)
        cameraOrient = self.fillOrientation(img.camera_orientation)
        obs.valid = True
        obs.timestamp = img.time_stamp
        obs.cameraImageU8 = cv2.cvtColor(cv2.imdecode(AirSimClient.stringToUint8Array(img.image_data_uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        obs.cameraPosition.x = cameraPos[b'x_val']
        obs.cameraPosition.y = cameraPos[b'y_val']
        obs.cameraPosition.z = cameraPos[b'z_val']
        obs.cameraRotation.pitch = cameraOrient[0]
        obs.cameraRotation.roll = cameraOrient[1]
        obs.cameraRotation.yaw = cameraOrient[2]
        obs.hasCollided = self.collInfo.has_collided

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
        print('------------')
        print('colinfo: %s'%self.collInfo.time_stamp)
        print('img: %s'%self.imgs[0].time_stamp)
        # if np.abs(diff) > 1e-4:
        #     printError('Timestamp difference between observation is too great : %s [s]'%(diff))
        #     return False
        #
        # Fill in all of the data.
        self.fillObservations(obs)
        return True
    # #
    # # Do the observation.
    # def observe(self, obs):
    #     return super(observations.Observer(), self).observe(obs)
