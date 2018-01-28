#!/usr/bin/env python3
import argparse
import SigHandler
from debug import *
import CuratorGui
import CuratorData
import sys
import ratelimiter
from PyQt5.QtWidgets import QApplication
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Tool used to mark usable and unusable data.')
    parser.add_argument('--path', dest='curationPath', default=None, type=str, help='Where to read data from for curation.')
    args = parser.parse_args()
    return args
#
# Rate limited stepping code limit to 30hz.
# @ratelimiter.RateLimiter(max_calls=30, period=1)
def step(gui):
    gui.visualize()
#
# Main loop for running the agent.
def loop():
    data = CuratorData.CuratorData(args.curationPath)
    gui = CuratorGui.CuratorGui(data)
    exitNow = SigHandler.SigHandler()
    while not exitNow.exit:
        step(gui)
#
# Main code.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    args = getInputArgs()
    loop()
