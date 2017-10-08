import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import Dataset
from debug import *

class Trainer():
    ''' Implements training neural networks.
        adapted from: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    '''

    def __init__(self, conf):
        ''' Set the training criteria.
        '''
        #
        # Save preferences.
        self.conf = conf
        hyperparam = self.conf.hyperparam
        self.model = hyperparam.model
        self.numEpochs = hyperparam.numEpochs
        self.criteria = hyperparam.criteria
        self.optimizer = hyperparam.optimizer
        self.bestModel = self.model.state_dict()
        self.bestAcc = 0
        #
        # Setup the dataset.
        train = Dataset.Dataset(conf, conf.dataTrainList, conf.transforms)
        self.dataloaders = {
        'train': torch.utils.data.DataLoader(train, batch_size = conf.hyperparam.batchSize, num_workers = conf.numWorkers, shuffle = True,  pin_memory = True),
        }
        #
        # No validation data, no need to evaluate it.
        if conf.dataValList != None and len(conf.dataValList) > 0:
            test = Dataset.Dataset(conf, conf.dataValList, conf.transforms)
            self.dataloaders['val'] = torch.utils.data.DataLoader(val, batch_size = conf.hyperparam.batchSize, num_workers = conf.numWorkers, shuffle = True,  pin_memory = True)

    def train(self):
        ''' Trains a neural netowork according to specified criteria.
        '''
        for epoch in range(self.numEpochs):
            printColor('Epoch {}/{}'.format(epoch, self.numEpochs - 1), colours.HEADER)
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
                for data in self.dataloaders[phase]:
                    inputs, labels = data['img'], data['labels']
                    if self.conf.usegpu:
                        labels =  labels.type(torch.LongTensor)[:,-1] #!!!remove this!!!
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    #
                    # Backward pass.
                    self.optimizer.zero_grad()
                    out = self.model(inputs)
                    _, preds = torch.max(out.data, 1)
                    loss = self.criteria(out, labels)
                    #
                    #  Backwards pass.
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    #
                    #  Stats.
                    runningLoss += loss.data[0]
                    runningCorrect += torch.sum(preds == labels.data)
                #
                # Overall stats.
                epochLoss = runningLoss / len(self.dataloaders[phase])
                epochAcc = runningCorrect / len(self.dataloaders[phase])
            #
            # Print per epoch results.
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epochLoss, epochAcc))
            #
            # Check if we have the new best model.
            if phase == 'val' and epochAcc > self.bestAcc:
                self.bestAcc = epochAcc
                self.bestModel = self.model.state_dict()
            #
            # Copy back the best model.
            self.model.load_state_dict(self.bestModel)
        printColor('Epochs complete!', colours.OKBLUE)
        return self.model
