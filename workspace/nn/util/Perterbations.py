import torch
from torchvision import transforms, utils
from PIL import ImageOps
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
            self.outputSize = (outputSize, outputSize)
        else:
            assert len(outputSize) == 2
            self.outputSize = outputSize

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']
        img_w, img_h = image.size
        #
        # check if the image to crop is large enough
        if self.outputSize[0] > img_h or self.outputSize[1] > img_w:
            printError("The image cannot be cropped because the image is too small. Model Image Shape:"+str(self.outputSize)+" Image Given:"+str(image.size))
            raise RuntimeError
        #
        # Crop the image to size.
        h_0 = int((img_h - self.outputSize[0])/2)
        w_1 = int((img_w - self.outputSize[1])/2)
        image = image.crop((w_1, h_0, w_1 + self.outputSize[0], h_0 + self.outputSize[1]))
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

    def __init__(self, outputSize, shiftBounds,  nSteps, mode = 'train'):
        self.tform = self.randomShift
        if mode is not 'train':
            self.tform = CenterCrop(outputSize)
        #
        # Output size.
        if isinstance(outputSize, int):
            self.outputSize = (outputSize, outputSize)
        else:
            assert len(outputSize) == 2
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
        self.shiftsx, self.stepx = np.linspace(-self.bounds[0], self.bounds[0], num = self.nBinsX, retstep=True)
        self.shiftsy, self.stepy = np.linspace(-self.bounds[1], self.bounds[1], num = self.nBinsY, retstep=True)
        #
        # If there is no delta, numpy returns NaN.
        if np.isnan(self.stepx):
            self.stepx = 0
        if np.isnan(self.stepy):
            self.stepy = 0
        #
        # In the case where there is only one step it sticks to the first side
        # of the range, instead we force the middle number to be zero.
        self.midIdxX = int(self.nBinsX / 2)
        self.midIdxY = int(self.nBinsY / 2)
        self.shiftsx[self.midIdxX] = 0
        self.shiftsy[self.midIdxY] = 0

    def __call__(self, sample):
        return self.tform(sample)

    def getShiftBounds(self):
        return self.shiftsx, self.shiftsy

    def randomShift(self, sample):
        image, labels = sample['img'], sample['labels']
        img_w, img_h = image.size
        bnd_x, bnd_y = self.bounds
        #
        # Check if the image to crop is large enough
        if self.outputSize[0] > img_h or self.outputSize[1] > img_w:
            printError("The image cannot be cropped because the image is too small. Model Image Shape:" + str(self.outputSize) + " Image Given:" + str(image.size))
            raise RuntimeError
        #
        # Check if we shifted outside of the bounds.
        if img_w < (2 * bnd_x + self.outputSize[0]) or img_h < (2 * bnd_y + self.outputSize[1]):
            printError("The image cannot be cropped because the image is too small. Model Image Shape:" + str(self.outputSize) + "Bounds:" + str(self.bounds)+ " Image Given:"+str(image.size))
            raise RuntimeError
        #
        # Select the shift.
        ix = random.randint(-self.rangeX, self.rangeX)
        iy = random.randint(-self.rangeY, self.rangeY)
        #
        # Randomly select a bin to shift the image based on the label, then add
        # additional noise to the shift. The additional variance is to help to
        # better generialize shifts within a single bin.
        dx = round(self.shiftsx[self.midIdxX + ix]) + random.randint(-round(self.stepx / 2), round(self.stepx / 2))
        dy = round(self.shiftsy[self.midIdxY + iy]) + random.randint(-round(self.stepy / 2), round(self.stepy / 2))
        #
        # Crop the image to size.
        h_0 = int((img_h - self.outputSize[0])/2 + dy)
        w_0 = int((img_w - self.outputSize[1])/2 + dx)
        image = image.crop((w_0, h_0, w_0 + self.outputSize[0], h_0 + self.outputSize[1]))
        #
        # Create the mask for where the correct values will exist when
        # flattened. Times 2 to account for there being a binary classification.
        mask = np.zeros((2 * self.nBinsX * self.nBinsY), dtype='int_')
        locX = self.midIdxX - ix
        locY = self.midIdxY - iy
        idx = 2 * (locY * self.nBinsX + locX)
        mask[idx:idx+2] = 1
        if self.nSteps == (0,0):
            mask = np.array([], dtype='int_')
        sample['meta']['shift'] = (dx,dy)
        #
        # Create the label dict, placing the mask within the label as the second
        # half.
        return {'img': image,
                'labels': np.append(labels[0], mask),
                'meta': sample['meta']}

class ColourJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, mode='train'):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.train = mode == 'train'

    def __call__(self, sample):
        img = sample['img']
        if self.train:
            t = transforms.ColorJitter.get_params(self.brightness, self.contrast,
                                                  self.saturation, self.hue)
            img = t(img)
        return {'img': img,
                'labels': sample['labels'],
                'meta': sample['meta']}

class RandomHorizontalFlip(object):
    """Randomly rotate an image between the specified angles.

    Args:
        flipProb (float): Probability of flipping the image horizontally.
    """
    def __init__(self, flipProb, mode='train'):
        self.flipProb = flipProb
        self.train = mode == 'train'

    def __call__(self, sample):
        labels = sample['labels']
        mask = labels[1:]
        im = sample['img']
        if self.train:
            if random.random() < self.flipProb:
                im = ImageOps.mirror(im)
                #
                # Flip the labels of the mask to match the flipped image.
                dx, dy = sample['meta']['shift']
                dx *= -1
                dy *= -1
                sample['meta']['shift'] = (dx, dy)
                mask = np.flip(mask, 0)
        return {'img': im,
                'labels': np.append(labels[0], mask),
                'meta': sample['meta']}

class RandomRotation(object):
    """Randomly rotate an image between the specified angles.

    Args:
        deg (float): Extents of the random rotation.
    """
    def __init__(self, deg, mode='train'):
        #
        # TODO: Possibly extend this to work for a larger range and choose the closest correct grid location.
        self.t = transforms.RandomRotation(deg)
        self.train = mode == 'train'

    def __call__(self, sample):
        img = sample['img']
        if self.train:
            img = self.t(img)
        return {'img': img,
                'labels': sample['labels'],
                'meta': sample['meta']}