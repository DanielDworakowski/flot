#!/usr/bin/env python
import DataLoader
import argparse
import Trainer
from debug import *
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
    configuration = __import__(args.configStr)
    conf = configuration.Config()
    #
    # Modifications to the configuration happen here.
    return conf
#
# Main loop for running the agent.
def train(conf):
    train = Trainer.Trainer()
#
# Main code.
args = getInputArgs()
conf = getConfig(args)
train(conf)
