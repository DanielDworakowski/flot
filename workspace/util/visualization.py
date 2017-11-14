#!/usr/bin/env python
import FlotDataset
import DefaultNNConfig
import torch
from DataUtil import plotSample
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import argparse
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import os
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Create visualization based on a network.')
    parser.add_argument('--config', dest='configStr', default='DefaultNNConfig', type=str, help='Name of the config file to import.')
    args = parser.parse_args()
    return args
#
# Get the configuration, override as needed.
def getConfig(args):
    configuration = __import__(args.configStr)
    conf = configuration.Config()
    #
    # Modifications to the configuration happen here.
    # conf.useTensorBoard = args.useTensorBoard
    return conf
#
# Gather data.
def gatherResponses(conf):
    train = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataset = torch.utils.data.DataLoader(train, batch_size = 32, num_workers = 1,
                                          shuffle = True, pin_memory = False)
    writer = SummaryWriter()
    loss = None
    probStack = None
    labelsStack = None
    model = conf.hyperparam.model
    model.train(False)
    sm = torch.nn.Softmax()
    meta = None
    #
    # Take a random sample from the dataset.
    for idx, data in enumerate(dataset):
        out = None
        labels = None
        if conf.usegpu:
            labels = Variable(data['labels'].long())
            labels =  labels.type(torch.LongTensor)[:,-1].cuda(async = True)
            out = conf.hyperparam.model(Variable(data['img']).cuda(async = True))
        else:
            out = conf.hyperparam.model(Variable(data['img']))

        probStack = sm(out).data
        labelsStack = data['labels'].long()
        meta = data['meta']
        break

    labelVec = torch.zeros(probStack.size())
    labelVec.scatter_(1, labelsStack, 1)
    diff = probStack.cpu() - labelVec
    dist = torch.sum(torch.mul(diff, diff), dim=1)

    sortedList, idx = torch.sort(dist)
    numImg = len(meta['index'])
    sidel = int(math.sqrt(numImg))
    f, axarr = plt.subplots(sidel,sidel)
    for x in range(sidel):
        for y in range(sidel):
            if (x*y) > numImg:
                break
            idx = x * sidel + y
            imName = os.path.join(meta['filedir'][idx], '%s_%s.png'%(conf.imgName, int(meta['index'][idx])))
            im = Image.open(imName)
            axarr[x,y].imshow(im)
            axarr[x,y].axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    # plotSample()
    plt.show()
    writer.close()


#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    conf = getConfig(args)
    gatherResponses(conf)
