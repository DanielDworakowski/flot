#!/usr/bin/env python
import pandas as pd 
import numpy as np
import argparse

def labellingParam():
    #
    # All the params are normalized to one. e.g 0.5 == 50%
    #
    # throwaway buffer for start and end of trajectory
    start_throwaway_buffer = 0.1
    end_throwaway_buffer = 0.1
    #
    # good and bad sections of the trajectory
    good_buffer = 0.3
    bad_buffer = 0.3
    #
    # middle throwaway buffer
    middle_throwaway_buffer = 1. - start_throwaway_buffer - end_throwaway_buffer - good_buffer - bad_buffer
    #
    # minimum trajectory length
    min_traj_len = 10
    return (start_throwaway_buffer, end_throwaway_buffer, good_buffer, bad_buffer, middle_throwaway_buffer, min_traj_len)

#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Auto labelling script via collision data')
    parser.add_argument('--obs', dest='observationsPath', default=None, type=str, help='Directory to the observations')
    args = parser.parse_args()
    return args
#
# Auto label data from collision information
def labelData(args):
    start_throwaway_buffer, end_throwaway_buffer, good_buffer, bad_buffer, middle_throwaway_buffer, min_traj_len = labellingParam()
    observations = pd.read_csv(args.observationsPath)
    observations = observations.rename(columns=lambda x: x.strip())
    collision_data = observations["raw_collision"].values
    col_idx = np.squeeze(np.argwhere(collision_data==1))
    trajs = np.split(collision_data, col_idx)
    labels = np.array([])
    for traj in trajs:
        traj_len = traj.shape[0]
        percentage = start_throwaway_buffer
        start_throwaway_idx = int(traj_len*percentage)
        percentage += good_buffer
        good_idx = int(traj_len*percentage)
        percentage += middle_throwaway_buffer
        middle_throwaway_idx = int(traj_len*percentage)
        percentage += bad_buffer
        bad_idx = int(traj_len*percentage)
        if traj.shape[0] < min_traj_len:
            traj[:] = -1
        else:
            traj[:start_throwaway_idx] = -1
            traj[start_throwaway_idx:good_idx] = 1
            traj[good_idx:middle_throwaway_idx] = -1
            traj[middle_throwaway_idx:bad_idx] = 0
            traj[bad_idx:] = -1
        labels=np.append(labels,traj)
    dataset = pd.DataFrame(data=labels, columns=["label"])
    labels_path = args.observationsPath.replace("observations.csv","labels.csv")
    dataset.to_csv(labels_path)

#
# Main code.
if __name__ == "__main__":
    args = getInputArgs()
    labelData(args)
