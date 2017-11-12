#!/usr/bin/env python
import FlotDataset
import DefaultNNConfig
import torch
from DataUtil import plotSample
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import argparse
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
    conf.useTensorBoard = args.useTensorBoard
    return conf
#
# Gather data.
def gatherResponses(args):



#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    conf - getConfig(args)
    gatherResponses(conf)
    #
    # The default configuration.
    conf = DefaultNNConfig.Config()
    train = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataset = torch.utils.data.DataLoader(train, batch_size = 32, num_workers = 1,
                                          shuffle = True, pin_memory = False)
    writer = SummaryWriter()
    probability = None
    labelsStack = None
    softmax_prob = torch.nn.Softmax()
    for data in dataset:
        out = None
        if self.conf.usegpu:
            out = conf.hyperparam.model(Variable(data['img']).cuda(async = True))
        else:
            out = conf.hyperparam.model(Variable(data['img']))

        labels = data['labels']
        if not probability:
            probability = softmax_prob(out).data.cpu()
            labelsStack = data['labels']
        else:
            torch.stack((probability,softmax_prob(out).data.cpu()))
            torch.stack(labelsStack,data['labels'])


        # for i in range (labels.size()[0]):
        #     if (probability[0][i][0] < probability [0][i][0]) == labels
        # max_index, val = torch.max(probability,1)
        print ("out")
        print(out)
        #print(torch.max(probability,1))
        print ("probability")
        print(probability)
        break


    writer.close()
