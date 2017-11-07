import numpy as np
import os
from abc import ABC, abstractmethod
from debug import *
import cv2
#
# Class defining a position vector.
class Vec3():
    #
    # Constructor.
    x = np.float32(0)
    y = np.float32(0)
    z = np.float32(0)
    #
    # Order in string format.
    def getFormat(self):
        return ('x[m], y[m], z[m]')
    #
    # Get the values into string format.
    def serialize(self):
        return ('%s, %s, %s')%(self.x, self.y, self.z)
#
# Class defining a rotation with euler angles.
class RotationEuler():
    #
    # Constructor.
    pitch = np.float32(0)
    roll = np.float32(0)
    yaw = np.float32(0)
    #
    # Order in string format.
    def getFormat(self):
        return ('roll[deg], pitch[deg], yaw[deg]')
    #
    # Get the values into string format.
    def serialize(self):
        return ('%s, %s, %s')%(self.roll, self.pitch, self.yaw)
#
# Class defining a compressed image and how to:
#   1. Expand it to a numpy array.
#   2. Serialize it.
class CompressedImage():
    #
    # Constructor.
    def __init__(self, name = 'front_camera', path = ''):
        self.path = path
        self.name = name
        self.pngImgs = None
        self.uint8Img = None
        self.path = path
        self.idx = 0
    #
    # Decompress the image into a unit8.
    def decompressPNG(self):
        if len(self.pngImgs) > 1:
            printWarn('More than one image may not be decompressing the correct type.')
        img = self.pngImgs[0]
        self.uint8Img = cv2.cvtColor(cv2.imdecode(np.fromstring(img.image_data_uint8, np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        return self.uint8Img
    #
    # String format for consistency.
    def getFormat(self):
        return ''
    #
    # Write a data string to file
    @staticmethod
    def writeToFile(filename, binData):
        with open(filename, 'wb') as f:
            f.write(binData)
    #
    # Serialize the image.
    def serialize(self):
        #
        # Error cases.
        if self.pngImgs == None or len(self.pngImgs) == 0:
            printError('No image data to save.')
            return ''
        if len(self.pngImgs) > 1:
            printWarn('More than one image may not be saving the correct type.')
        img = self.pngImgs[0]
        path = '%s/%s_%s'%(self.path, self.name, self.idx)
        if img.pixels_as_float:
            print("Type %d, size %d" % (img.image_type, len(img.image_data_float)))
            printError('Saving float images is not supported.')
        else:
            print("Type %d, size %d" % (img.image_type, len(img.image_data_uint8)))
            print(path)
            self.writeToFile(os.path.normpath(path + '.png'), img.image_data_uint8)
        return ''
#
# Class defining a generic serializable object.
class GenericObservation():
    #
    # Constructor.
    def __init__(self, name):
        self.name = name
        self.val = None
    #
    # String format for header.
    def getFormat(self):
        return '%s'%(self.name)
    #
    # Generic serialization.
    def serialize(self):
        if self.val == None:
            printError('Observation has no value.')
            self.val = m.nan
        return '%s'%(self.val)
#
# Class defining a generic set of observations.
class Observation():
    #
    # Constructor.
    def __init__(self, path):
        self.valid = False
        self.serializable = {
            'idx': GenericObservation('idx'),
            'timestamp_ns': GenericObservation('timestamp_ns'),
            'cameraPosition': Vec3(),
            'cameraRotation': RotationEuler(),
            'hasCollided': GenericObservation('raw_collision'),
            'img': CompressedImage(path = path),
        }
        #
        # Keep track of how many lines have been saved.
        self.serializable['idx'].val = 0
    #
    # Get a particular observation by key.
    def __getitem__(self, key):
        return self.serializable[key]
    #
    # String format for header.
    def getFormat(self):
        fmt = ''
        sep = ', '
        for key in self.serializable:
            names = self.serializable[key].getFormat()
            #
            # Not all data is serialized into a csv.
            if names == '':
                continue
            fmt += self.serializable[key].getFormat() + sep
        #
        # Remove the ', '.
        fmt = fmt[:-len(sep)]
        return fmt
    #
    # Serialize the object.
    def serialize(self):
        line = ''
        sep = ', '
        if self.valid:
            for key in self.serializable:
                data = self.serializable[key].serialize()
                #
                # Not all data is serialized into a csv.
                if data == '':
                    continue
                line += data + sep
            self.serializable['idx'].val += 1
            self.serializable['img'].idx = self.serializable['idx'].val
            return line[:-len(sep)]
        else:
            printError('Unable to serialize data.')
            return ''

#
# Abstract class defining the interface of an observer.
class Observer():
    #
    # Initialization.
    def __init__(self, obsDir = ''):
        self.obsDir = obsDir
        self.obsCsv = None
        self.obs = Observation(obsDir)
    #
    # Open observation file.
    def __enter__(self):
        self.obsCsv = open(self.obsDir + '/observations.csv', 'w')
        self.obsCsv.write(self.obs.getFormat() + '\n')
        return self
    #
    # Close the observation file.
    def __exit__(self, type, value, traceback):
        if self.obsCsv:
            self.obsCsv.close()
            self.obsCsv = None
        return False
    #
    # Fills and returns observations about the environment.
    @abstractmethod
    def observeImpl(self):
        pass
    #
    # Do the observation.
    def observe(self):
        return self.observeImpl(self.obs)
    #
    # Serialize the data.
    # Since the data is in a standardized format there is no need to have an
    # abstraction.
    def serialize(self):
        ln = self.obs.serialize()
        if ln == '':
            return False
        self.obsCsv.write(ln + '\n')
