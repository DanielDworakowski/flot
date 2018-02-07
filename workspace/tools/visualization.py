#!/usr/bin/env python
import os
import sys
import math
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from nn.core import FlotDataset
from nn.util import Perterbations
from torch.autograd import Variable
from nn.config import DefaultNNConfig
from tensorboardX import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageFont, ImageDraw
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Create visualization based on a network.')
    parser.add_argument('--config', dest='configStr', default='DefaultNNConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('--batchSize', dest='bsize', default=25, type=int, help='How many images to put in a batch')
    parser.add_argument('--plotVisited', dest='pltVisited', default=False,  help='Plot what was visited by the blimp', action='store_true')
    parser.add_argument('--plotBatch', dest='pltBatch', default=False, help='Plot and order the responses of a random batch', action='store_true')
    parser.add_argument('--plotMT', dest='pltMeanTraj', default=False, help='Plot the mean trajectories.', action='store_true')
    parser.add_argument('--watch', dest='watch', default=False,  help='Watch some of the plots grow up', action='store_true')
    parser.add_argument('--useLabels', dest='uselabels', default=False,  help='Use the labels file instead of the observations files for vis', action='store_true')
    args = parser.parse_args()
    return args
#
# Get the configuration, override as needed.
def getConfig(args):
    config_module = __import__('config.' + args.configStr)
    configuration = getattr(config_module, args.configStr)
    conf = configuration.Config('test')
    return conf
#
# Generate a lookup table for a colour gradient from r -> b gradients
def rtobTable():
    rng = 255
    r,g,b = 255, 20, 0
    dr = -1.
    dg = 0.
    db = 1.
    rgb = []
    for i in range(rng):
        r,g,b = r+dr, g+dg, b+db
        rgb.append((int(r), int(g), int(b)))
    return rgb
#
# Draw trajectory dots on the image.
def drawTrajectoryDots(x, y, space, sidel, rgbTable, draw, conf, posClassProb):
    nBin = posClassProb.shape[0]
    #
    # Super lazy way to not have to repeat calculations.
    shiftBoundsx, shiftBoundsy = Perterbations.RandomShift(conf.hyperparam.image_shape, conf.hyperparam.shiftBounds, conf.hyperparam.nSteps).getShiftBounds()
    lenX = len(shiftBoundsx)
    for idx in range(nBin):
        yIdx = int(idx / lenX)
        xIdx = idx - yIdx * lenX
        dx = shiftBoundsx[xIdx]
        dy = shiftBoundsy[yIdx]
        x_dot = round(x + sidel[0] / 2 + dx)
        y_dot = round(y + sidel[1] / 2 + dy)
        prob = posClassProb[idx]
        colour = rgbTable[int(255 * prob)]
        draw.ellipse((x_dot - space / 2 , y_dot - space / 2, x_dot + space / 2, y_dot + space / 2), fill = colour)
#
# Plot the image results from best to worst.
def plotBatch(conf, meta, sorted, batchinfo):
    numImg = len(meta['index'])
    sidel = int(math.ceil(math.sqrt(numImg)))
    imNames = meta['imName']
    testim = imNames[0]
    im = Image.open(testim)
    width, height = im.size
    inW, inH = conf.hyperparam.image_shape
    grid = Image.new('RGB', (inW * sidel, inH * sidel))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("sans-serif.ttf", 25)
    fillarr = ['red', 'blue']
    dx, dy = 0, 0
    rgbTable = rtobTable()
    for i, y in enumerate(range(0, inW * sidel, inW)):
        for j, x in enumerate(range(0, inH * sidel, inH)):
            infoIdx = i * sidel + j
            if infoIdx >= numImg:
                break
            idx = sorted[i * sidel + j]
            imName = imNames[idx]
            #
            # TODO: Check if this actually shifts the visualization image properly.
            if 'shift' in meta:
                dx, dy = meta['shift'][0][idx], meta['shift'][1][idx]
            im = Image.open(imName)
            im = im.crop(
                (
                    width  / 2  - inW / 2 + dx,
                    height / 2  - inH / 2 + dy,
                    width  / 2  + inW / 2 + dx,
                    height / 2  + inH / 2 + dy
                )
            )
            grid.paste(im, (x,y))
            space = 15
            draw.ellipse((x, y, x + space, y + space), fill=fillarr[batchinfo['labels'][idx][0]])
            _, predicted = torch.max(batchinfo['probs'][idx], 0)
            draw.ellipse((x + space, y, x + 2 * space, y + space), fill=fillarr[predicted[0]])
            if type(batchinfo['posClasses']) != type(None):
                drawTrajectoryDots(x, y, space / 2, im.size, rgbTable, draw, conf, batchinfo['posClasses'][idx, :])
            draw.text((x + 3 * space,y),'%s'%(i*sidel + j),(255,255,255),font=font)
            draw.text((x,y + space),'%1.2f'%batchinfo['probs'][idx][1],(255,255,255),font=font)
    implt = plt.figure()
    impltax = implt.gca()
    impltax.imshow(grid)
    impltax.axis('off')
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
    histax.set_xlabel('Label')
    histax.set_ylabel('Frequency')
    numBins = 2
    histData = np.ones(len(dataset))
    #
    # Iteratively create the plots.
    colour = ['red', 'blue']
    if args.watch:
        plt.ion()
    for idx, data in enumerate(dataloader):
        if args.watch:
            plt.pause(0.00001)
        allLabels = data['meta']['allLabels']
        for miniIdx in range(len(allLabels['x[m]'])):
            axScatter.scatter(allLabels['x[m]'][miniIdx], allLabels['y[m]'][miniIdx], color=colour[data['labels'][miniIdx][0]])
            histidx = idx * args.bsize + miniIdx
            histData[histidx] = allLabels['collision_free'][miniIdx]
    histax.hist(histData, numBins)
    histax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    histax.set_xticklabels(('','Negative', '', 'Positive',''))
#
# Gather and plot the trajectory lengths.
def plotMeanTraj(args, conf):
    allTrajs = []
    for dir in conf.dataTrainList:
        distl = getMeanTrajSingle(args, conf, dir)
        allTrajs.extend(distl)
    allTrajs = np.array(allTrajs)
    # allTrajs = allTrajs[allTrajs > 1]
    print('Mean traj length')
    print(np.mean(allTrajs))
    print('Std dev traj length')
    print(np.std(allTrajs))
    figHist = plt.figure()
    histax = figHist.gca()
    numBins = int(math.sqrt(len(allTrajs)))
    histax.hist(allTrajs, numBins)
#
# Get the mean trajectory length in the data.
def getMeanTrajSingle(args, conf, dir):
    file = 'observations.csv'
    if args.uselabels:
        file = 'labels.csv'
    observations = pd.read_csv(dir + '/' + file)
    observations = observations.rename(columns=lambda x: x.strip())
    x_idx = observations.columns.get_loc('x[m]')
    y_idx = observations.columns.get_loc('y[m]')
    z_idx = observations.columns.get_loc('z[m]')
    collision_data = observations["raw_collision"].values
    col_idx = np.squeeze(np.argwhere(collision_data==1))
    trajs = np.array_split(observations.as_matrix(), col_idx)
    distList = []
    #
    # Calculate the lengths of each of the trajectories
    for traj in trajs:
        if len(traj) < 40:
            continue
        traj_x = traj[:, x_idx]
        traj_y = traj[:, y_idx]
        traj_z = traj[:, z_idx]
        traj_x_shift = np.delete(np.roll(np.copy(traj_x), -1), -1)
        traj_y_shift = np.delete(np.roll(np.copy(traj_y), -1), -1)
        traj_z_shift = np.delete(np.roll(np.copy(traj_z), -1), -1)
        traj_x = np.delete(traj_x,-1)
        traj_y = np.delete(traj_y,-1)
        traj_z = np.delete(traj_z,-1)
        dist = np.sum(np.sqrt(np.power(traj_x - traj_x_shift, 2) + np.power(traj_y - traj_y_shift, 2) + np.power(traj_z - traj_z_shift, 2)))
        distList.append(dist)
    return distList
#
# Gather data.
def gatherResponses(conf, dataloader):
    loss = None
    probStack = None
    labelsStack = None
    model = conf.hyperparam.model
    model.train(False)
    sm = torch.nn.Softmax(dim = 1)
    meta = None
    numCorrect = 0
    posClasses = None
    #
    # Take a random sample from the dataloader.
    for idx, data in enumerate(dataloader):
        out = None
        labels = None
        if conf.usegpu:
            labels = Variable(data['labels'].squeeze_()).cuda(async = True)
            out = conf.hyperparam.model(Variable(data['img']).cuda(async = True))
        else:
            out = conf.hyperparam.model(Variable(data['img']))
        #
        # Check if the model can return the postive class probability.
        if hasattr(conf.hyperparam.model, 'getClassifications'):
            posClasses = conf.hyperparam.model.getClassifications(out, sm)
        #
        # Check if we need to use the class to get gather activations.
        if hasattr(conf.hyperparam.model, 'getActivations'):
            out, labels = conf.hyperparam.model.getActivations(out, labels)
            labels = labels.data
        else:
            labels = data['labels']
        probStack = sm(out).data
        _, preds = torch.max(out.data, 1)
        labelsStack = labels.long()
        meta = data['meta']
        numCorrect += torch.sum(preds == labels.cuda(async = True))
        epochAcc = numCorrect / (preds.size()[0])
        print(epochAcc)
        break

    labelVec = torch.zeros(probStack.size())
    labelsStack = torch.unsqueeze(labelsStack, 1)
    labelVec.scatter_(1, labelsStack.cpu(), 1)
    diff = probStack.cpu() - labelVec
    dist = torch.sum(torch.mul(diff, diff), dim=1)
    #
    # Get the sorted version of the batch.
    sortedList, idx = torch.sort(dist)
    figHist = plt.figure()
    histax = figHist.gca()
    histax.set_xlabel('Score (Lower is better)')
    histax.set_ylabel('Frequency')
    histax.hist(dist.numpy(), int(math.ceil(math.sqrt(dist.size()[0])))) # Dont do sqrt so up to 2.
    return {'probs': probStack, 'meta': meta, 'sorted': idx, 'metric': dist, 'labels': labelsStack, 'posClasses': posClasses}
#
# Visualize.
def visualize(conf, args):
    dataset = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.bsize, num_workers = 8,
                                          shuffle = True, pin_memory = False)
    if args.pltBatch:
        batchinfo = gatherResponses(conf, dataloader)
        plotBatch(conf, batchinfo['meta'], batchinfo['sorted'], batchinfo)
    if args.pltVisited:
        plotTrajectory(args, conf, dataloader, dataset)
    if args.pltMeanTraj:
        plotMeanTraj(args, conf)
# 
# View a tensor.
def showTensor(tensor):
    from torchvision import transforms
    vt = transforms.ToPILImage()
    newimg = vt(data['img']).show()
#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    conf = getConfig(args)
    if conf.modelLoadPath == None or conf.modelLoadPath == '':
        print('No model was loaded cannot perform visualization!')
        sys.exit()
    visualize(conf, args)
    plt.show()
