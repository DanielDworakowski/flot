#
# Built in.
import os
import time
import shutil
#
# Torch.
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
#
# Other.
import tqdm
import numpy as np
from tensorboardX import SummaryWriter
#
# Flot.
from debug import *
from itertools import count
from core import FlotDataset
#
# Speeds things up??
torch.backends.cudnn.benchmark = True

class Trainer():
    ''' Implements training neural networks.
        adapted from: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    '''
    def __setupDatasets(self):
        ''' Setups up datasets from configuration.
        '''
        train = FlotDataset.FlotDataset(self.conf, self.conf.dataTrainList, self.conf.transforms)
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(train, batch_size = self.conf.hyperparam.batchSize, num_workers = self.conf.numWorkers, shuffle = True,  pin_memory = True),
        }
        #
        # No validation data, no need to evaluate it.
        if self.conf.dataValList != None and len(self.conf.dataValList) > 0:
            test = FlotDataset.FlotDataset(self.conf, self.conf.dataValList, self.conf.transforms)
            self.dataloaders['val'] = torch.utils.data.DataLoader(test, batch_size = self.conf.hyperparam.batchSize, num_workers = self.conf.numWorkers, shuffle = True,  pin_memory = True)

    def __setupLogging(self):
        ''' Configuration for logging the training process.
        '''
        #
        # Setup tensorboard as require.
        def doNothing(self, tmp = None):
            pass
        #
        # Run every epoch.
        def logEpochTensorboard(self, epochSummary):
            self.logger.add_scalar('%s_loss'%epochSummary['phase'], epochSummary['loss'], epochSummary['epoch'])
            self.logger.add_scalar('%s_acc'%epochSummary['phase'], epochSummary['acc'], epochSummary['epoch'])
            labels = None
            if len(epochSummary['data']['labels'].shape) > 1:
                labels = epochSummary['data']['labels'][:, 0]
            else:
                labels = epochSummary['data']['labels']
            for i in range(epochSummary['data']['labels'].shape[0]):
                self.logger.add_image('{}_image_i-{}_epoch-{}_pre-:{}_label-{}'.format(epochSummary['phase'], i, epochSummary['epoch'], epochSummary['pred'][i], int(labels[i])), epochSummary['data']['img'][i], epochSummary['epoch'])
            for name, param in self.model.named_parameters():
                self.logger.add_histogram(name, param.clone().cpu().data.numpy(), epochSummary['epoch'])
        #
        # Write everything as needed.
        def closeTensorboard(self):
            self.logger.close()
        #
        # Setup functions.
        self.logEpoch = doNothing
        self.closeLogger = doNothing
        #
        # Change defaults.
        if self.conf.useTensorBoard:
            self.logger = SummaryWriter()
            self.logEpoch = logEpochTensorboard
            self.closeLogger = closeTensorboard

    def __saveCheckpoint(self, epoch, isBest):
        ''' Save a model.
        '''
        state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'model': self.model,
                    'conf': self.conf
                }
        savePath = '%s/%s_epoch_%s.pth.tar'%(self.conf.modelSavePath, time.strftime('%d-%m-%Y-%H-%M-%S'), epoch)
        torch.save(state, savePath)
        if isBest:
            shutil.move(savePath, '%s/%s_model_best.pth.tar'%(self.conf.modelSavePath, conf.experimentName))

    def __init__(self, conf):
        ''' Set the training criteria.
        '''
        #
        # Save preferences.
        self.conf = conf
        hyperparam = self.conf.hyperparam
        self.model = hyperparam.model
        self.numEpochs = hyperparam.numEpochs
        self.batchSize = hyperparam.batchSize
        self.criteria = hyperparam.criteria
        self.optimizer = hyperparam.optimizer
        self.bestModel = self.model.state_dict()
        self.bestAcc = 0
        self.startingEpoch = 0
        self.logger = None
        self.logEpoch = None
        self.closeLogger = None
        self.model = conf.hyperparam.model
        self.optimizer = conf.hyperparam.optimizer
        self.startingEpoch = conf.startingEpoch
        self.__setupDatasets()
        self.__setupLogging()

    def train(self):
        ''' Trains a neural netowork according to specified criteria.
        '''
        for epoch in range(self.startingEpoch, self.startingEpoch + self.numEpochs):
            printColour('Epoch {}/{}'.format(epoch, self.startingEpoch + self.numEpochs - 1), colours.HEADER)
            for phase in self.dataloaders:
                #
                # Switch on / off gradients.
                self.model.train(phase == 'train')
                #
                # The current loss.
                runningLoss = 0.0
                runningCorrect = 0.0
                #
                # Iterate over data.
                numMini = len(self.dataloaders[phase])
                pbar = tqdm.tqdm(total=numMini)
                for data in self.dataloaders[phase]:
                    inputs, labels_cpu = data['img'], data['labels']
                    if self.conf.usegpu:
                        labels_cpu.squeeze_()
                        inputs, labels = Variable(inputs).cuda(async = True), Variable(labels_cpu).cuda(async = True)
                    else:
                        inputs, labels = Variable(inputs), Variable(labels_cpu)
                    #
                    # Forward through the model and optimize.
                    out = self.model(inputs)
                    preds, loss, dCorrect = self.model.pUpdate(self.optimizer, self.criteria, out, labels, data['meta'], phase)
                    #
                    #  Stats.
                    runningLoss += loss.data[0]
                    runningCorrect += dCorrect
                    pbar.update(1)
                pbar.close()
                #
                # Overall stats
                epochLoss = runningLoss / len(self.dataloaders[phase])
                epochAcc = runningCorrect / (len(self.dataloaders[phase])*self.batchSize)
                #
                # Check if we have the new best model.
                isBest = False
                if phase == 'val' and epochAcc > self.bestAcc:
                    isBest = True
                    self.bestAcc = epochAcc
                    self.bestModel = self.model.state_dict()
                #
                # Print per epoch results.
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epochLoss, epochAcc))
                summary = {
                    'phase': phase,
                    'epoch': epoch,
                    'loss': epochLoss,
                    'acc': epochAcc,
                    'data': data,
                    'pred' : preds
                }
                self.logEpoch(self, summary)
                #
                # Save model as needed.
                if ((epoch % self.conf.epochSaveInterval) == 0 and phase == 'train') or isBest:
                    self.__saveCheckpoint(epoch, isBest)
        #
        # Copy back the best model.
        self.model.load_state_dict(self.bestModel)
        printColour('Epochs complete!', colours.OKBLUE)
        self.closeLogger(self)
        return self.model
