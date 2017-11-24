#!/usr/bin/env python
import argparse
import Trainer
from debug import *
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General tool to train a NN based on passed configuration.')
    parser.add_argument('--config', dest='configStr', default='DefaultNNConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('--useTB', dest='useTensorBoard', default=False,  action='store_true', help='Whether to create a tensorboard visualization.')
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
# Main loop for running the agent.
def train(conf):
    train = Trainer.Trainer(conf)
    train.train()
#
# Main code.
if __name__ == '__main__':
    args = getInputArgs()
    conf = getConfig(args)
    train(conf)
