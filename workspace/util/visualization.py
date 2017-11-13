#!/usr/bin/env python
import FlotDataset
import DefaultNNConfig
import torch
from DataUtil import plotSample
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import argparse
import numpy as np
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
    #
    # Iterate through the dataset.
    for idx, data in enumerate(dataset):
        out = None
        labels = None
        if conf.usegpu:
            labels = Variable(data['labels'].long())
            labels =  labels.type(torch.LongTensor)[:,-1].cuda(async = True)
            out = conf.hyperparam.model(Variable(data['img']).cuda(async = True))
        else:
            out = conf.hyperparam.model(Variable(data['img']))

        if probStack is None:
            probStack = sm(out).data
            labelsStack = data['labels'].long()
        else:
            torch.stack((probStack,sm(out).data))
            torch.stack((labelsStack,data['labels'].long()))


        # # for i in range (labels.size()[0]):
        # #     if (probability[0][i][0] < probability [0][i][0]) == labels
        # # max_index, val = torch.max(probability,1)
        # print ("out")
        # print(out)
        # #print(torch.max(probability,1))
        # print ("probability")
        print(probStack)
        # print(labels)
        # break
    labelVec = torch.zeros(probStack.size())
    labelVec.scatter_(1, labelsStack, 1)
    diff = probStack.cpu() - labelVec
    dist = torch.sum(torch.mul(diff, diff), dim=1)

    writer.close()


#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    conf = getConfig(args)
    gatherResponses(conf)
