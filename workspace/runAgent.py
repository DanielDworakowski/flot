#!/usr/bin/env python3
import sys
import argparse
import util.SigHandler as SigHandler
import environment.Environment as Environment
import ratelimiter
from debug import *
import environment.Observations as Observations
import util.AgentVisualization as visual
from PyQt5.QtWidgets import QApplication

# Image uploading
import cv2
import numpy as np
import requests
addr = 'http://10.0.1.49:5000/updateImage'
content_type = 'image/jpeg'
headers = {'content-type': content_type}

#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General running environment for simulation and real hardware.')
    parser.add_argument('--path', dest='dataColPath', default=None, type=str, help='Where to save the data.')
    parser.add_argument('--agent', dest='agentStr', default=None, type=str, help='Name of the agent file to import.')
    parser.add_argument('--config', dest='configStr', default='RobotConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('--serialize', dest='serialize', default=None, type=bool, help='Whether to serialize training data.')
    parser.add_argument('--rateLimit', dest='ratelimit', default=False,  action='store_true', help='Whether to serialize training data.')
    args = parser.parse_args()
    return args
args = getInputArgs()
#
# Get the configuration, override as needed.
def getConfig(args):
    conf = __import__(args.configStr).Config()
    if args.dataColPath != None:
        conf.savePath = args.dataColPath
    if args.agentStr != None:
        conf.agentType = args.agentStr
    if args.serialize != None:
        conf.serialize = args.serialize
    conf.getAgentConstructor()
    return conf
#
# Conditional decorator.
# https://stackoverflow.com/questions/20850571/decorate-a-function-if-condition-is-true
def maybe_decorate(condition, decorator):
    return decorator if condition else lambda x: x
#
# Rate limited stepping code limit to 30hz.
@maybe_decorate(args.ratelimit,ratelimiter.RateLimiter(max_calls=30, period=1))
def step(agent, env, vis):
    obs = env.observe()
    agent.giveObservation(obs)
    action = agent.getAction()
    env.runAction(action, obs)
    vis.visualize(obs, action, agent)
#
# Main loop for running the agent.
def loop(conf):
    agent = conf.agentConstructor(conf)
    # vis = visual.Visualizer()
    exitNow = SigHandler.SigHandler()

    with Environment.Environment(conf.envType, conf.getFullSavePath(conf.serialize), conf.serialize) as env:

        while not exitNow.exit:
            step(agent, env, vis)
            #
            # Streaming to server endpoint via POST requests
            # try:
            #     frame = env.observer.stream.getFrame()
            #     if frame is not None:
            #         _, img_encoded = cv2.imencode('.jpg', frame)
            #         response = requests.post(addr, data=img_encoded.tostring(), headers=headers)
            # except Exception as e:
            #     pass
# Main code.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    conf = getConfig(args)
    vis = visual.Visualizer()

    loop(conf)
