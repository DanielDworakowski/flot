#!/usr/bin/env python
from core import FlotDataset
from config import DefaultNNConfig
from nn.util import DataUtil
import tqdm
from torchvision import transforms
import nn.util.DataUtil as DataUtil
import nn.util.Perterbations as Perterbations
import torch
import argparse
import numpy as np
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
# Get images statistics.
def getStatistics(conf):
    n1 = 128
    n2 = 0
    #
    # Change the transformations to only crop to the correct size.
    t = transforms.Compose([
        Perterbations.CenterCrop(conf.hyperparam.cropShape),
        DataUtil.Rescale(conf.hyperparam.image_shape),
        DataUtil.ToTensor(),
    ])
    dataset = FlotDataset.FlotDataset(conf, conf.dataTrainList, t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = n1, num_workers = conf.numWorkers, shuffle = False,  pin_memory = False)
    #
    # Iterate over the dataset and obtain individual means and std-devs.

# CURMEAN  [ 0.18943557  0.21603064  0.21423657]
# curVAR  [ 0.03896314  0.04170484  0.04159307]
    totMean = 0
    totVar = 0
    numMini = len(dataloader)
    pbar = tqdm.tqdm(total=numMini)
    for data in dataloader:
        imgs, labels_cpu = data['img'], data['labels']
        curMean = np.mean(imgs.numpy(), axis=(0,2,3))
        curVar = np.sqrt(np.var(imgs.numpy(), axis=(0,2,3)))
        oldMean = totMean
        totMean = (n1 * curMean + n2 * totMean) / (n1 + n2)
        totVar = (n1 * curVar + n2 * totVar + n1 * np.power((curMean - totMean), 2) + n2 * np.power((oldMean - totMean), 2)) / (n1 + n2)
        n2 += n1
        pbar.update(1)
    pbar.close()
    #
    # Print the final results.
    print('Mean of the data: %s'%totMean)
    print('stddev of the data: %s'%totVar)



if __name__ == '__main__':
    #
    # Obtain the conficguration.
    args = getInputArgs()
    conf = getConfig(args)
    stats = getStatistics(conf)

