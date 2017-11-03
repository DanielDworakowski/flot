#!/usr/bin/env python3
import Environment
import Observations
import argparse
import SigHandler
import ratelimiter
from debug import *
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General running environment for simulation and real hardware.')
    parser.add_argument('--path', dest='dataColPath', default=None, type=str, help='Where to save the data.')
    parser.add_argument('--agent', dest='agentStr', default=None, type=str, help='Name of the agent file to import.')
    parser.add_argument('--config', dest='configStr', default='DefaultConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('--serialize', dest='serialize', default=None, type=bool, help='Whether to serialize training data.')
    args = parser.parse_args()
    return args
#
# Get the configuration, override as needed.
def getConfig(args):
    configuration = __import__(args.configStr)
    conf = configuration.Config()
    if args.dataColPath != None:
        conf.savePath = args.dataColPath
    if args.agentStr != None:
        conf.agentType = args.agentStr
    if args.serialize != None:
        conf.serialize = args.serialize
    conf.getAgentConstructor()
    return conf
#
# Rate limited stepping code limit to 30hz.
@ratelimiter.RateLimiter(max_calls=30, period=1)
def step(agent, env):
    obs = env.observe()
    agent.giveObservation(obs)
    action = agent.getAction()
    env.runAction(action, obs)
#
# Main loop for running the agent.
def loop(conf):
    agent = conf.agentConstructor(conf)
    exitNow = SigHandler.SigHandler()
    with Environment.Environment(conf.envType, conf.getFullSavePath(), conf.serialize) as env:
        while not exitNow.exit:
            step(agent, env)
#
# Main code.
if __name__ == "__main__":
    args = getInputArgs()
    conf = getConfig(args)
    loop(conf)
