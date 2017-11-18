#!/usr/bin/env python
import argparse
import os
import subprocess
import random
#
# https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Generate data lists by spliting testing and training data.')
    parser.add_argument('--baseDir', dest='baseDir', default=None, type=str, help='Base directory to check for data.', required=True)
    parser.add_argument('--split', dest='split', default=0.2, type=float, help='What  is validation.')
    args = parser.parse_args()
    return args
#
# Traverse the data directories.
def traverseDirs(args):
    base = os.path.abspath(args.baseDir)
    splitf = args.split
    allData = []
    tdata = 0
    tFiles = 0
    for root, dirs, files in os.walk(base):
        if 'labels.csv' in files:
            nData = file_len(root+'/labels.csv')
            if nData < 2:
                continue
            allData.append({'ndata': nData, 'dir': root})
            tdata += nData
            tFiles += len(files)
    numValFiles = splitf * tdata
    valData = []
    tval = 0
    while tval < numValFiles:
        takeIdx = int(random.random() * len(allData))
        tval += allData[takeIdx]['ndata']
        valData.append(allData.pop(takeIdx))
    print('Training data:\n-----------------------------------------')
    for name in allData:
        print('\'%s/\','%name['dir'])
    print('Validation data:\n-----------------------------------------')
    for name in valData:
        print('\'%s/\','%name['dir'])
    print('Final split : %f'%(tval / tdata))
#
# Main code.
if __name__ == '__main__':
    args = getInputArgs()
    traverseDirs(args)
