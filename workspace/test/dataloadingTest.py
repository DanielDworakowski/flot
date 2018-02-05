#!/usr/bin/env python
from core import FlotDataset
from config import DefaultNNConfig
from nn.util import DataUtil
from tensorboardX import SummaryWriter
import torch
import argparse
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General tool to train a NN based on passed configuration.')
    parser.add_argument('--config', dest='configStr', default='DefaultNNConfig', type=str, help='Name of the config file to import.')
    args = parser.parse_args()
    return args
#
# Get the configuration, override as needed.
def getConfig(args):
    config_module = __import__('config.' + args.configStr)
    configuration = getattr(config_module, args.configStr)
    conf = configuration.Config()
    #
    # Modifications to the configuration happen here.
    return conf
# 
# Main.
if __name__ == '__main__':
    #
    # The default configuration.
    conf = getConfig(getInputArgs())
    train = FlotDataset.FlotDataset(conf, conf.dataTrainList, conf.transforms)
    dataset = torch.utils.data.DataLoader(train, batch_size = 32, num_workers = 1,
                                          shuffle = True, pin_memory = False)
    writer = SummaryWriter()
    for data in dataset:
        if torch.cuda.is_available():
            writer.add_image('Image', data['img'].cuda(), 0)
        else:
            writer.add_image('Image', data['img'], 0)
        writer.add_text('Text', 'text logged at step:'+str(1), 1)
        DataUtil.plotSample(data)
        break
    writer.close()
