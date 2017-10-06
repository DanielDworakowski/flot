from __future__ import print_function, division
from debug import *
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from interval_tree import IntervalTree

class DataLoader(Dataset):
    '''Read from a list all of the data files.'''

    def __init__(self, conf, pathList, transform):
        '''
        Args:

        '''
        self.conf = conf
        self.dataList = []
        self.offsets = []
        self.len = 0
        features = []
        for idx, dataFolder in enumerate(pathList):
            feature = []
            self.dataList.append(DataFolder(conf, dataFolder, transform))
            feature.append(self.len)
            self.offsets.append(self.len)
            self.len += len(self.dataList[-1])
            feature.append(self.len)
            feature.append(idx)
            features.append(feature)
        self.binSelector = IntervalTree(features, 0, self.len + 1)

    def __len__(self):
        '''
        Args:

        '''
        return self.len

    def __getitem__(self, idx):
        binIdx = self.binSelector.find_range([idx, idx])
        if binIdx == None:
            printError('selected impossible index %s', idx)
            return None
        return self.dataList[binIdx].__getitem__[idx - self.offsets[idx]]

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

    def __getitem__(self, idx):
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
