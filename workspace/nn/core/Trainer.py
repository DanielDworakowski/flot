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
        self.bestModel = model.state_dict()
        self.bestAcc = 0
        #
        # Setup the dataset.
        train = DataLoader.DataLoader(conf, conf.dataTrainList, conf.transforms)
        test = DataLoader.DataLoader(conf, conf.dataValList, conf.transforms)
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(train, batch_size = conf.hyperparam.batchSize, num_workers = conf.numWorkers, shuffle = True,  pin_memory = True),
            'val': torch.utils.data.DataLoader(val, batch_size = conf.hyperparam.batchSize, num_workers = conf.numWorkers, shuffle = True,  pin_memory = True),
        }
        #
        # No validation data, no need to evaluate it.
        if self.dataLoaders['val'] == None or len(self.dataLoaders['val']) == 0:
            del self.dataLoaders['val']

    def train(self):
        ''' Trains a neural netowork according to specified criteria.
        '''
        for epoch in range(self.numEpochs):
            printColor('Epoch {}/{}'.format(epoch, num_epochs - 1), colours.HEADER)
            for phase in ['train', 'val']:
                #
                # Switch on / off gradients.
                model.train(phase == 'train')
                #
                # The current loss.
                runningLoss = 0.0
                runningCorrect = 0.0
                #
                # Iterate over data.
                for data in self.dataloaders[phase]:
                    inputs, labels = data
                    if self.conf.useGpu:
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    #
                    # Begin forwarding process.
                    self.optimizer.zero_grad()
                    outs = self.model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)
                    #
                    #  Backwards pass.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    #
                    #  Stats.
                    runningLoss += loss.data[0]
                    runningCorrect += torch.sum(preds == labels.data)
                #
                # Overall stats.
                epochLoss = runningLoss / len(self.dataloaders[phase])
                epochLoss = runningCorrect / len(self.dataloaders[phase])
            #
            # Print per epoch results.
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epochLoss, epochAcc))
            #
            # Check if we have the new best model.
            if phase == 'val' and epoch_acc > best_acc:
                self.bestAcc = epoch_acc
                self.bestModel = self.model.state_dict()
            #
            # Copy back the best model.
            model.load_state_dict(self.bestModel)
            return model
