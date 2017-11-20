#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse
import os

def labellingParam():
    #
    # All the params are normalized to one. e.g 0.5 == 50%
    # throwaway buffer for start and end of trajectory
    start_throwaway_buffer = 0.4
    end_throwaway_buffer = 0.0
    #
    # good and bad sections of the trajectory
    good_buffer = 0.35
    bad_buffer = 0.1
    #
    # middle throwaway buffer
    middle_throwaway_buffer = 1. - start_throwaway_buffer - end_throwaway_buffer - good_buffer - bad_buffer
    #
    # minimum trajectory length
    min_traj_dist = 3
    return (start_throwaway_buffer, end_throwaway_buffer, good_buffer, bad_buffer, middle_throwaway_buffer, min_traj_dist)

#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Auto labelling script via collision data')
    parser.add_argument('--obs', dest='observationsPath', nargs='+', default=None, type=str, help='Full path to the obervations csv.')
    parser.add_argument('--obsFolders', dest='folderPath', nargs='+', default=None, type=str, help='Folder that contains multiple data folders')
    args = parser.parse_args()
    return args
#
# Auto label data from collision information
def labelData(observationsPath):
    start_throwaway_buffer, end_throwaway_buffer, good_buffer, bad_buffer, middle_throwaway_buffer, min_traj_dist = labellingParam()
    try:
        observations = pd.read_csv(observationsPath)
    except:
        print('Unable to open %s'%observationsPath)
        return
    observations = observations.rename(columns=lambda x: x.strip())
    collision_data = observations["raw_collision"].values
    col_idx = np.squeeze(np.argwhere(collision_data==1))
    x_idx = observations.columns.get_loc('x[m]')
    y_idx = observations.columns.get_loc('y[m]')
    z_idx = observations.columns.get_loc('z[m]')
    try:
        trajs = np.array_split(observations.as_matrix(), col_idx)
        col_trajs = np.split(collision_data, col_idx)
    except:
        print('No collisions detected in file %s'%observationPath)
        return
    labels = np.array([])
    for i in range(len(col_trajs)):
        traj = trajs[i]
        col_traj = col_trajs[i]
        traj_len = col_traj.shape[0]
        percentage = start_throwaway_buffer
        start_throwaway_idx = int(traj_len*percentage)
        percentage += good_buffer
        good_idx = int(traj_len*percentage)
        percentage += middle_throwaway_buffer
        middle_throwaway_idx = int(traj_len*percentage)
        percentage += bad_buffer
        bad_idx = int(traj_len*percentage)

        # Calculating the trajectory distance
        traj_x = traj[:, x_idx]
        traj_y = traj[:, y_idx]
        traj_z = traj[:, z_idx]
        traj_x_shift = np.delete(np.roll(np.copy(traj_x), -1), -1)
        traj_y_shift = np.delete(np.roll(np.copy(traj_y), -1), -1)
        traj_z_shift = np.delete(np.roll(np.copy(traj_z), -1), -1)
        traj_x = np.delete(traj_x,-1)
        traj_y = np.delete(traj_y,-1)
        traj_z = np.delete(traj_z,-1)
        dist = np.sum(np.sqrt(np.power(traj_x - traj_x_shift, 2) + np.power(traj_y - traj_y_shift, 2) + np.power(traj_z - traj_z_shift, 2)))

        if dist < min_traj_dist:
            col_traj[:] = -1
        else:
            col_traj[:start_throwaway_idx] = -1
            col_traj[start_throwaway_idx:good_idx] = 1
            col_traj[good_idx:middle_throwaway_idx] = -1
            col_traj[middle_throwaway_idx:bad_idx] = 0
            col_traj[bad_idx:] = -1
        labels=np.append(labels,col_traj)
    observations.insert(1,'collision_free',labels)
    dataset = observations
    dataset = dataset[dataset.collision_free!=-1]
    labels_path = observationsPath.replace("observations.csv","labels.csv")
    dataset.to_csv(labels_path, index=False)

#
# Main code.
if __name__ == "__main__":
    args = getInputArgs()
    if args.observationsPath == None and args.folderPath == None:
        print('Must specify path to parse.')
    elif args.observationsPath != None:
        for observationsPath in args.observationsPath:
            labelData(os.path.abspath(observationsPath))
    else:
        observationsPath = [x[0]+'/observations.csv' for x in os.walk(args.folderPath[0])][1:]
        for observationPath in observationsPath:
            labelData(os.path.abspath(observationPath))
