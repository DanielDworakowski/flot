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
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Create visualization based on a network.')
    parser.add_argument('--config', dest='configStr', default='DefaultNNConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('--batchSize', dest='bsize', default=32, type=int, help='How many images to put in a batch')
    parser.add_argument('--plotVisited', dest='pltVisited', default=False,  help='Plot what was visited by the blimp', action='store_true')
    parser.add_argument('--plotBatch', dest='pltBatch', default=False, help='Plot and order the responses of a random batch', action='store_true')
    parser.add_argument('--plotMT', dest='pltMeanTraj', default=False, help='Plot the mean trajectories.', action='store_true')
    parser.add_argument('--watch', dest='watch', default=False,  help='Watch somee of the plots grow up', action='store_true')
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
def gatherResponses(conf, dataloader):
    loss = None
    probStack = None
    labelsStack = None
    model = conf.hyperparam.model
    model.train(False)
    sm = torch.nn.Softmax()
    meta = None
    #
    # Take a random sample from the dataloader.
    for idx, data in enumerate(dataloader):
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
# make matplot lib plots.
def plotTrajectory(args, conf, dataloader, dataset):
    mpl.rcParams['legend.fontsize']= 14
    fig = plt.figure()
    #
    # Scatter plot for the visited locations.
    axScatter = plt.gca()
    axScatter.set_xlabel('X')
    axScatter.set_ylabel('Y')
    fighist = plt.figure()
    histax = fighist.gca()
    numBins = 2
    histData = np.ones(len(dataset))
    #
    # Iteratively create the plots.
    if args.watch:
        plt.ion()
    for idx, data in enumerate(dataloader):
        if args.watch:
            plt.pause(0.00001)
        allLabels = data['meta']['allLabels']
        for miniIdx in range(len(allLabels['x[m]'])):
            axScatter.scatter(allLabels['x[m]'][miniIdx], allLabels['y[m]'][miniIdx])
            histidx = idx * args.bsize + miniIdx
            histData[histidx] = allLabels['collision_free'][miniIdx]
    histax.hist(histData, numBins)
    plt.show()
#
# Visualize.
def visualize(conf, args):
    dataset = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.bsize, num_workers = 8,
                                          shuffle = True, pin_memory = False)
    # batchinfo = gatherResponses(conf, dataloader)
    # plotBatch(conf, batchinfo['meta'], batchinfo['sorted'], batchinfo)
    plotTrajectory(args, conf, dataloader, dataset)
#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    conf = getConfig(args)
    visualize(conf, args)
