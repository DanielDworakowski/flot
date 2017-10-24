# from __future__  import print_function, division
from debug import *
from queue import Empty,Full,Queue
from torch.autograd import Variable
from interval_tree import IntervalTree
from torchvision import transforms, utils
from skimage import io, transform, img_as_float
import os
import time
import torch
import threading
import numpy as np
import pandas as pd
import torch.utils.data


class CudaDataLoader(threading.Thread):
    """Read a cpudataloader and automatically push data to CUDA."""

    def __init__(self, queue, cpuDataLoader, useGPU):
        threading.Thread.__init__(self)
        self.running = True
        self.queue = queue
        self.cpuDataLoader = cpuDataLoader
        self.useGPU = useGPU

    def startThread(self):
        """ Start the thread """
        self.start()

    def stop(self):
        """ Exit the thread """
        self.running = False

    def joinThread(self):
        """ Join the thread """
        self.join()

    def run(self):
        while self.running:
            for data in self.cpuDataLoader:
                inputs, labels = data['img'], data['labels']
                #
                # Send to the GPU as needed.
                if self.useGPU:
                    labels =  labels.type(torch.LongTensor)[:, -1] #!!!remove this!!!]
                    inputs, labels = Variable(inputs.cuda(async = True)), Variable(labels.cuda(async = True))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                self.queue.put((inputs, labels), block = True)


class FlotDataset(torch.utils.data.Dataset):
    """Read from a list all of the data files."""

    def __init__(self, conf, pathList, transform):
        """
        Args:

        """
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
        """
        Args:

        """
        return self.len

    def __getitem__(self, idx):

        binIdx = self.binSelector.find_range([idx, idx])[0]
        if binIdx == None:
            printError('selected impossible index %s', idx)
            return None
        return self.dataList[binIdx].__getitem__(idx - self.offsets[binIdx])

class DataFolder(torch.utils.data.Dataset):
    """Read from a data folder."""

    def __init__(self, conf, dataPath, transform = None):
        """
        Args:

        """
        self.csvFileName = conf.csvFileName
        self.rootDir = dataPath
        self.transform = transform
        self.csvFrame = pd.read_csv(self.rootDir + '/' + self.csvFileName)
        self.imgColIdx = self.csvFrame.columns.get_loc('idx')
        self.conf = conf

    def __len__(self):
        """
        Args:

        """
        return len(self.csvFrame)

    def __getitem__(self, idx):
        """
        Args:

        """

        labels = self.csvFrame.ix[idx]
        imName = os.path.join(self.rootDir, '%s_%s.png'%(self.conf.imgName, int(labels[self.imgColIdx])))
        img = io.imread(imName) #/ np.float32(255)
        #
        # Remove the column index.
        labels = np.delete(labels.as_matrix(), self.imgColIdx)
        sample = {'img': img, 'labels': labels}
        #
        # Transform as needed.
        if self.transform:
            sample = self.transform(sample)

        return sample
