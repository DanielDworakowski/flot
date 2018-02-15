#!/usr/bin/env python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Create visualization based on a network.')
    parser.add_argument('--runs', dest='runs', default='runFolder', type=str, help='Directory to the runs folder.')
    parser.add_argument('--smooth', dest='alpha', default=0, type=float, help='Exponential moving average smoothing alpha.')
    args = parser.parse_args()
    return args
#
# Exponential moving avg.
def numpy_ewma_vectorized_v2(data, alpha):
    alpha_rev = 1-alpha
    n = data.shape[0]
    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out
#
# Plot.
def gatherData(path, trainlax, trainAax, vallax, valAax, name, alpha):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags()['scalars'])
    if len(event_acc.Tags()['scalars']) == 0:
        return

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    _, trainlstp, trainLoss = zip(*event_acc.Scalars('train_loss'))
    _, trainvstp, trainAcc = zip(*event_acc.Scalars('train_acc'))
    _, vallstp, valLoss = zip(*event_acc.Scalars('val_loss'))
    _, trainvstp, valAcc = zip(*event_acc.Scalars('val_acc'))
    trainLoss = np.array(trainLoss)
    trainAcc = np.array(trainAcc)
    valAcc = np.array(valAcc)
    valLoss = np.array(valLoss)
    trainLoss = numpy_ewma_vectorized_v2(trainLoss, alpha)
    trainLoss = numpy_ewma_vectorized_v2(trainAcc, alpha)
    valAcc = numpy_ewma_vectorized_v2(valAcc, alpha)
    valLoss = numpy_ewma_vectorized_v2(valLoss, alpha)
    trainlax.plot(trainlstp, trainLoss, label=name)
    trainAax.plot(trainvstp, trainAcc, label=name)
    vallax.plot(vallstp, valLoss, label=name)
    valAax.plot(trainvstp, valAcc, label=name)
#
# Iterate over directories.
def doPath(args):
    trainl = plt.figure()
    trainA = plt.figure()
    vall = plt.figure()
    valA = plt.figure()
    trainlax = trainl.gca()
    trainlax.set_xlabel('Epoch')
    trainlax.set_ylabel('Loss')
    trainlax.set_title('Training loss')
    trainAax = trainA.gca()
    trainAax.set_xlabel('Epoch')
    trainAax.set_ylabel('Accuracy (%)')
    trainAax.set_title('Training Accuracy')
    vallax = vall.gca()
    vallax.set_xlabel('Epoch')
    vallax.set_ylabel('Loss')
    vallax.set_title('Validation Loss')
    valAax = valA.gca()
    valAax.set_xlabel('Epoch')
    valAax.set_ylabel('Accuracy (%)')
    valAax.set_title('Validation Accuracy')
    base = os.path.abspath(args.runs)
    for root, dirs, files in os.walk(base):
        dirs.sort()
        for direc in (dirs):
            print(direc)
            gatherData(root+'/'+direc, trainlax, trainAax, vallax, valAax, direc, args.alpha)
    trainl.savefig('trainl.png')
    vallax.legend(loc='lower left')
    valAax.legend(loc='lower left')
    trainlax.legend(loc='lower left')
    trainAax.legend(loc='lower left')
    trainA.savefig('trainA.png')
    vall.savefig('vall.png')
    valA.savefig('valA.png')

    plt.show()


#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    doPath(args)
