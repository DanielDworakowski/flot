from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class DataFolder(Dataset):
    '''Read from a data folder.'''

    def __init__(self, conf, dataPath, transform = None):
        '''
        Args:

        '''
        self.csvFileName = conf.csvFileName
        self.rootDir = rootDir
        self.transform = transform
        self.csvFrame = pd.read_csv(rootDir + '/' + csvFileName)

    def __len__(self):
        '''
        Args:

        '''
        return len(self.csvFrame)

    def __getitem____(self, idx):
        '''
        Args:

        '''
        imName = os.path.join(self.rootDir, self.imgName + '_' + idx + '.png')
        img = io.imread(imName)
        labels = self.csvFrame.ix[idx, 1:]
        sample = {'img': img, 'labels': labels}
        #
        # Transform as needed.
        if self.transform:
            sample = self.transform(sample)
        return sample
