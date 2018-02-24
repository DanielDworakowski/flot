#!/usr/bin/env python

import argparse
import os
import pandas as pd
import numpy as np

def getInputArgs():
    parser = argparse.ArgumentParser('Postprocessing for blimp')
    parser.add_argument('--file', dest='dataPath', default=None, type=str, help='~/.ros/<the name of this file>')
    args = parser.parse_args()
    return args

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def main(d):
    outname = 'out.csv'
    path = os.path.abspath(d)
    files = listdir_fullpath(path)
    pngs = []
    csvs = []

    for x in files:
        if x.endswith('csv') and not x.endswith(outname):
            csvs.append(x)
        elif x.endswith('png'):
            pngs.append(x)

    dfs = {}
    video_ts = None
    for csv in csvs:
        df = pd.read_csv(csv)
        if 'video_ts' in csv:
            df.columns = ['Timestamp']
            df.rename(index=str, columns={'':'Timestamp'})
            df = df.iloc[1:] / 1e3
            video_ts = df.iloc[1:]
        dfs[csv.split('.')[0].split('/')[-1]] = df

    if type(video_ts) is None:
        print('Cannot process this folder as there is no data file for the timestamps.')
        sys.exit(0)

    pngs = sorted(pngs)
    # png_times = sorted([float(x.split('/')[-1].rsplit('.',1)[0]) for x in pngs])
    png_times = video_ts
    # print(png_times.keys())
    # png_times.loc[:, 'Timestamp'] = png_times['Timestamp'] / 1e3

    for k, df in dfs.items():
        df = df.rename(columns=lambda x: x.strip())
        dfs[k] = df.sort_values('Timestamp')

    arrmin = []
    arrmax = []
    for df in dfs.values():
        arrmin.append(min(df['Timestamp'].tolist()))
        arrmax.append(max(df['Timestamp'].tolist()))

    lowerbound = max(arrmin)
    upperbound = min(arrmax)
    counts = []

    for k, df in dfs.items():
        df = df.drop(df[df.Timestamp < lowerbound].index)
        df = df.drop(df[df.Timestamp > upperbound].index)
        df = df.reset_index(drop=True)
        counts.append(len(df.index))
        dfs[k] = df

    sensory_dfs = dfs.copy()
    # print(dfs)
    png_time = sensory_dfs.pop('video_ts')

    png_list = np.array(png_time['Timestamp'].tolist())
    closest_values = {}
    for k, df in sensory_dfs.items():
        df_times = np.array(df['Timestamp'].tolist())
        png_rep = np.tile(png_list, (df_times.shape[0],1))
        df_times_rep = np.transpose(np.tile(df_times, (png_list.shape[0],1)))
        res = abs(png_rep - df_times_rep)
        idx = np.argmin(res, axis=0)
        closest_values[k] = idx

    l_ind = png_times[png_times.Timestamp < lowerbound].index
    u_ind = png_times[png_times.Timestamp > upperbound].index

    if not l_ind.empty and not u_ind.empty:
        png_out = pngs[max(l_ind)+1:min(u_ind)]
    elif not l_ind.empty:
        png_out = pngs[max(l_ind)+1:]
    elif not u_ind.empty:
        png_out = pngs[:min(u_ind)]
    else:
        png_out = pngs[:]

    l = []
    for k, df in sensory_dfs.items():
        tmp = df.loc[closest_values[k]]
        tmp = tmp.reset_index(drop=True)
        tmp = tmp.drop(['Timestamp'], axis=1)
        tmp = tmp.rename(columns=lambda x:'{}:{}'.format(k,x))
        l.append(tmp)

    png_out = sorted([x.split('/')[-1] for x in png_out])
    png_out = pd.DataFrame({'PNG':png_out})
    png_out = png_out.reset_index(drop=True)
    l.append(png_out)

    res = pd.concat(l, axis=1)
    res.to_csv('{}/out.csv'.format(path), index=False)


if __name__=='__main__':
    args = getInputArgs()
    main(args.dataPath)
