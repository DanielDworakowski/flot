#!/usr/bin/env python3
import sys
import argparse
import SigHandler
import CuratorGui
import CuratorData
from util.debug import *
from PyQt5.QtWidgets import QApplication
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Tool used to mark usable and unusable data.')
    parser.add_argument('--path', dest='curationPath', default=None, type=str, help='Where to read data from for curation.')
    parser.add_argument('--conf', dest='configStr', default=None, type=str, help='Configuration used to load a model.')
    args = parser.parse_args()
    return args
#
# Get the configuration, override as needed.
def getConfig(args):
    conf = None
    model = None
    if args.configStr != None:
        config_module = __import__('config.' + args.configStr)
        configuration = getattr(config_module, args.configStr)
        conf = configuration.Config('train')
        model = conf.hyperparam.model.eval()
    return conf, model
#
# Main loop for running the agent.
def loop(args, conf, model):
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    data = CuratorData.CuratorData(args.curationPath)
    data.autoLabel()
    gui = CuratorGui.CuratorGui(data)
    gui.setModel(model, conf)
    exitNow = SigHandler.SigHandler()
    while not exitNow.exit and gui.running:
        gui.visualize()
#
# Main code.
if __name__ == "__main__":
    args = getInputArgs()
    conf, model = getConfig(args)
    loop(args, conf, model)
