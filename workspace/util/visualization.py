#!/usr/bin/env python
import FlotDataset
import DefaultNNConfig
import torch
from DataUtil import plotSample
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Create visualization based on a network.')
    parser.add_argument('--config', dest='configStr', default='DefaultNNConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('--batchSize', dest='bsize', default=32, type=int, help='How many images to put in a batch')
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
# Plot the image results from best to worst.
def plotBatch(conf, meta, sorted, batchinfo):
    numImg = len(meta['index'])
    sidel = int(math.sqrt(numImg)) + 1
    testim =  os.path.join(meta['filedir'][0], '%s_%s.png'%(conf.imgName, int(meta['index'][0])))
    im = Image.open(testim)
    width, height = im.size
    grid = Image.new('RGB', (width * sidel, height * sidel))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("sans-serif.ttf", 25)
    for i, y in enumerate(range(0, width * sidel, width)):
        for j, x in enumerate(range(0, height * sidel, height)):
            infoIdx = i * sidel + j
            if infoIdx >= numImg:
                break
            idx = sorted[i * sidel + j]
            imName = os.path.join(meta['filedir'][idx], '%s_%s.png'%(conf.imgName, int(meta['index'][idx])))
            im = Image.open(imName)
            grid.paste(im, (x,y))
            draw.text((x,y),'%s'%(i*sidel + j),(255,255,255),font=font)
    plt.imshow(grid)
    plt.show()
#
# Gather data.
def gatherResponses(conf, bsize):
    train = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataset = torch.utils.data.DataLoader(train, batch_size = bsize, num_workers = 1,
                                          shuffle = True, pin_memory = False)
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
    #
    # Get the sorted version of the batch.
    sortedList, idx = torch.sort(dist)
    return {'meta': meta, 'sorted': idx, 'metric': dist}
#
# Visualize.
def visualize(conf, args):
    batchinfo = gatherResponses(conf, args.bsize)
    plotBatch(conf, batchinfo['meta'], batchinfo['sorted'], batchinfo)
#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    conf = getConfig(args)
    visualize(conf, args)
