import torch
from torchvision import transforms, utils
import random
import numpy as np
from debug import *

class CenterCrop(object):
    '''Crop the center of the image for training.

    Args:
        outputSize (tuple or int): Desired output size. If int, square crop
            is made.
    '''

    def __init__(self, outputSize):
        if isinstance(outputSize, int):
            self.outputSize = (outputSize, outputSize, 3)
        else:
            assert len(outputSize) == 3
            self.outputSize = outputSize

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']
        img_h, img_w, img_c = image.shape
        #
        # check if the image to crop is large enough
        if self.outputSize[0] > img_h or self.outputSize[1] > img_w:
            printError("The image cannot be cropped because the image is too small. Model Image Shape:"+str(self.outputSize)+" Image Given:"+str(image.shape))
            raise RuntimeError
        #
        # Crop the image to size.
        h_0 = int((img_h - self.outputSize[0])/2)
        w_1 = int((img_w - self.outputSize[1])/2)
        image = image[h_0:h_0+self.outputSize[0],w_1:w_1+self.outputSize[1],:]
        return {'img': image,
                'labels': labels,
                'meta': sample['meta']}

class RandomShift(object):
    ''' Take a random shift of the ROI of the image.

    Args:
        shiftBounds (tuple or int): Desired bounds of random shift. If int,
            bounds are assumed to be square. Bounds are treated as pixels.
        outputSize (tuple or int): Desired output size. If int, square crop
            is made.
        discretization (int): step size of the shifts (for example if 5, steps
            will be 0, 5, 10 etc.)

    '''

    def __init__(self, outputSize, shiftBounds,  nSteps):
        #
        # Output size.
        if isinstance(outputSize, int):
            self.outputSize = (outputSize, outputSize, 3)
        else:
            assert len(outputSize) == 3
            self.outputSize = outputSize
        #
        # Bounds for the shifting
        if isinstance(shiftBounds, int):
            self.bounds = (shiftBounds, shiftBounds)
        else:
            assert len(shiftBounds) == 2
            self.bounds = shiftBounds
        #
        # Bounds for discretization
        if isinstance(nSteps, int):
            self.nSteps = (nSteps, nSteps)
        else:
            assert len(nSteps) == 2
            self.nSteps = nSteps
        #
        # Setup the bounds and shifts.
        self.rangeX = nSteps[0]
        self.rangeY = nSteps[1]
        self.nBinsX = 2 * self.rangeX + 1
        self.nBinsY = 2 * self.rangeY + 1
        self.shiftsx = np.linspace(-self.bounds[0], self.bounds[0], num = self.nBinsX)
        self.shiftsy = np.linspace(-self.bounds[1], self.bounds[1], num = self.nBinsY)
        #
        # In the case where there is only one step it sticks to the first side
        # of the range, instead we force the middle number to be zero.
        self.midIdxX = int(self.nBinsX / 2)
        self.midIdxY = int(self.nBinsY / 2)
        self.shiftsx[self.midIdxX] = 0
        self.shiftsy[self.midIdxY] = 0

    def getShiftBounds(self):
        return self.shiftsx, self.shiftsy

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']
        img_h, img_w, img_c = image.shape
        bnd_x, bnd_y = self.bounds
        #
        # Check if the image to crop is large enough
        if self.outputSize[0] > img_h or self.outputSize[1] > img_w:
            printError("The image cannot be cropped because the image is too small. Model Image Shape:"+str(self.outputSize)+" Image Given:"+str(image.shape))
            raise RuntimeError
        #
        # Check if we shifted outside of the bounds.
        if img_w < (2 * bnd_x + self.outputSize[0]) or img_h < (2 * bnd_y + self.outputSize[1]):
            printError("The image cannot be cropped because the image is too small. Model Image Shape:"+str(self.outputSize)+ "Bounds:"+str(self.bounds)+ " Image Given:"+str(image.shape))
            raise RuntimeError
        #
        # Select the shift.
        ix = random.randint(-self.rangeX, self.rangeX)
        iy = random.randint(-self.rangeY, self.rangeY)
        dx = round(self.shiftsx[ix])
        dy = round(self.shiftsy[iy]) # Add noise around the bins?
        #
        # Crop the image to size.
        h_0 = int((img_h - self.outputSize[0])/2 + dy)
        w_1 = int((img_w - self.outputSize[1])/2 + dx)
        image = image[h_0:h_0+self.outputSize[0],w_1:w_1+self.outputSize[1],:]
        #
        # Create the mask for where the correct values will exist when
        # flattened. Times 2 to account for there being a binary classification.
        mask = np.zeros((2 * self.nBinsX * self.nBinsY), dtype='long')
        locX = self.midIdxX - ix
        locY = self.midIdxY - iy
        idx = 2 * (locY * self.nBinsX + locX)
        mask[idx:idx+2] = 1
        sample['meta']['shift'] = (dx,dy)
        #
        # Create the label dict, placing the mask within the label as the second
        # half.
        return {'img': image,
                'labels': np.append(labels[0], mask),
                'meta': sample['meta']}
