#!/usr/bin/env python
import numpy as np
from datetime import datetime
import argparse
import os
import importlib
from env_settings import env_settings
import time
import environment.AI2THOR as ai2thor
import torch
import util.MountainCarContinousPixel as mountain_env

# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('RL with AI2THOR')
    parser.add_argument('--phase', dest='phase', default='train', type=str, help='train or test')
    parser.add_argument('--algorithm', dest='algorithm', default='A2S', type=str, help='algorithm name')
    args = parser.parse_args()
    return args

def train(agent_class, id, env_name, seed, record, data_collection_params, training_params, network_params, algorithm_params, logs_path):
    env = mountain_env.Env()
    # env = ai2thor.AI2THOR(env_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = agent_class.Agent(env, sess, data_collection_params, training_params, network_params, algorithm_params, logs_path)
    try:
        os.mkdir(save_dir)
    except:
        pass
    agent.train(saver=saver, save_dir=save_dir)

if __name__ == "__main__":
    args = getInputArgs()
    env_setting = env_settings[args.algorithm]
    
    if args.phase == 'train':
        train(**env_setting)

    else:
        print("ERROR: INVALID ARGUMENT Please choose train or test for phase argument")