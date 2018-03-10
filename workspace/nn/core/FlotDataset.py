from __future__ import print_function, division
import os
import time
import torch
import functools
import numpy as np
from debug import *
import pandas as pd
from PIL import Image
import torch.utils.data
from interval_tree import IntervalTree
from torchvision import transforms, utils

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
    return newfunc

class FlotDataset(torch.utils.data.Dataset):
    '''Read from a list all of the data files.'''

    def __init__(self, conf, pathList, transform):
        self.conf = conf
        self.dataList = []
        self.offsets = []
        self.len = 0
        features = []
        for idx, dataFolder in enumerate(pathList):
            feature = []
            #
            # This can be done better by using pytorch to concat datasets.
            self.dataList.append(DataFolder(conf, dataFolder, transform))
            feature.append(self.len)
            self.offsets.append(self.len)
            self.len += len(self.dataList[-1])
            feature.append(self.len - 1)
            feature.append(idx)
            features.append(feature)
        self.binSelector = IntervalTree(features, 0, self.len + 1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        binIdx = self.binSelector.find_range([idx, idx])[0]
        if binIdx == None:
            printError('selected impossible index %s', idx)
            return None
        return self.dataList[binIdx].__getitem__(idx - self.offsets[binIdx])

class DataFolder(torch.utils.data.Dataset):
    '''Read from a data folder.'''

    def __init__(self, conf, dataPath, transform = None):
        self.csvFileName = conf.csvFileName
        self.rootDir = dataPath
        self.transform = transform
        self.csvFrame = pd.read_csv(self.rootDir + '/' + self.csvFileName)
        self.getImNameFn = None
        self.conf = conf
        #
        # If the full image name is already part of the label just take the image name.
        try:
            self.pngColIdx = self.csvFrame.columns.get_loc('PNG')
            self.getImNameFn = self.getImgNamePNG
        except:
            self.getImNameFn = self.getImgNameIdx
        self.imgColIdx = self.csvFrame.columns.get_loc('idx')
        #
        # Check if we need to setup our own labels.
        try:
            self.labelIdx = self.csvFrame.columns.get_loc('collision_free')
        except:
            self.csvFrame['collision_free'] = (self.csvFrame['Sonar:Smoothed'] > self.conf.distThreshold).astype(int)
            self.labelIdx = self.csvFrame.columns.get_loc('collision_free')
        if self.__len__() < 1:
            printError('File %s has no data. This will cause issues'%conf.csvFileName)
            raise ValueError

    def getImgNameIdx(self, labels):
        return os.path.join(self.rootDir, '%s_%s.png'%(self.conf.imgName, int(labels[self.imgColIdx])))

    def getImgNamePNG(self, labels):
        return os.path.join(self.rootDir, labels[self.pngColIdx])

    def __len__(self):
        return len(self.csvFrame)

    def __getitem__(self, idx):
        try:
            labels = self.csvFrame.ix[idx]
        except:
            printError('Indexing error')
            raise ValueError
        imName = self.getImNameFn(labels)
        #
        # Construct meta data.
        meta = {
            'filedir': self.rootDir,
            'imName': imName,
            'index': int(labels[self.imgColIdx]),
            'allLabels':  labels.to_dict(),
            'shift': (0,0)
        }
        img = Image.open(imName).convert('RGB')
        #
        # Remove the column index.
        # labels = np.delete(labels.as_matrix(), self.imgColIdx)
        labels = np.full((1,1),labels.as_matrix()[self.labelIdx], dtype='int_')
        sample = {'img': img, 'labels': labels, 'meta': meta}
        #
        # Transform as needed.
        if self.transform:
            sample = self.transform(sample)
        return sample
