#!/usr/bin/env python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import argparse
import matplotlib.pyplot as plt

#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Create visualization based on a network.')
    parser.add_argument('--runs', dest='runs', default='runFolder', type=str, help='Directory to the runs folder.')
    args = parser.parse_args()
    return args
def gatherData(path, trainlax, trainAax, vallax, valAax):
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
    trainlax.plot(trainlstp, trainLoss)
    trainAax.plot(trainvstp, trainAcc)
    vallax.plot(vallstp, valLoss)
    valAax.plot(trainvstp, valAcc)
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
    valAax.set_ylabel('Validation Accuracy')
    valAax.set_title('Validation Accuracy')

    base = os.path.abspath(args.runs)
    for root, dirs, files in os.walk(base):
        for direc in dirs:
            print(direc)
            gatherData(root+'/'+direc, trainlax, trainAax, vallax, valAax)
    trainl.savefig('trainl.png')
    trainA.savefig('trainA.png')
    vall.savefig('vall.png')
    valA.savefig('valA.png')

    plt.show()


#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    doPath(args)
