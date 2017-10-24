#
# Built in.
import time
import os
from queue import Empty,Full,Queue
#
# Torch.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
#
# Other.
from tensorboardX import SummaryWriter
import numpy as np
import tqdm
#
# Flot.
import FlotDataset
from debug import *

class Trainer():
    ''' Implements training neural networks.
        adapted from: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    '''

    def __setupDatasets(self):
        ''' Setups up datasets from configuration.
        '''
        #
        # Plain CPU dataloaders.
        train = FlotDataset.FlotDataset(self.conf, self.conf.dataTrainList, self.conf.transforms)
        self.dataQueues = {
            'train': Queue(maxsize = 128)
        }
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(train, batch_size = self.conf.hyperparam.batchSize, num_workers = self.conf.numWorkers, shuffle = True,  pin_memory = True),
        }
        self.cudaLoaders = {
            'train': FlotDataset.CudaDataLoader(self.dataQueues['train'], self.dataloaders['train'], self.conf.usegpu)
        }
        #
        # No validation data, no need to evaluate it.
        if self.conf.dataValList != None and len(self.conf.dataValList) > 0:
            val = FlotDataset.FlotDataset(self.conf, self.conf.dataValList, self.conf.transforms)
            self.dataloaders['val'] = torch.utils.data.DataLoader(val, batch_size = self.conf.hyperparam.batchSize, num_workers = self.conf.numWorkers, shuffle = True,  pin_memory = True)
            self.dataQueues['val'] = Queue(max = 128)
            self.cudaLoaders['val'] = FlotDataset.CudaDataLoader(self.dataQueues['val'], self.dataloaders['val'], self.conf.usegpu)

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
            for i in range(epochSummary['data']['labels'].shape[0]):
                self.logger.add_image('{}_image_i-{}_epoch-{}_pre-:{}_label-{}'.format(epochSummary['phase'],i,epochSummary['epoch'],epochSummary['pred'][i],int(epochSummary['data']['labels'][i].numpy()[0])), epochSummary['data']['img'][i], epochSummary['epoch'])
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

    def __loadModel(self):
        ''' Load model from a specified directory.
        '''
        if self.conf.modelLoadPath != None and os.path.isfile(self.conf.modelLoadPath):
            checkpoint = torch.load(self.conf.modelLoadPath)
            self.model = checkpoint['model']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.startingEpoch = checkpoint['epoch']
            printColour('Loaded model from path: %s'%self.conf.modelLoadPath, colours.OKBLUE)
        elif self.conf.modelLoadPath != None:
            printError('Unable to load specified model: %s'%(self.conf.modelLoadPath))


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
            shutil.copyfile(savePath, '%s/model_best.pth.tar'%(self.conf.modelSavePath))

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
        self.__loadModel() # must happen after loading of config.
        self.__setupDatasets()
        self.__setupLogging()

    def trainSerialLoader(self):
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
                    inputs, labels = data['img'], data['labels']
                    start = time.time()
                    if self.conf.usegpu:
                        labels =  labels.type(torch.LongTensor)[:, -1] #!!!remove this!!!]
                        inputs, labels = Variable(inputs.cuda(async = True)), Variable(labels.cuda(async = True))
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
                    pbar.update(1)
                    end = time.time()
                    print('Elapsed %dus'%((end-start)*1e6))
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
                if (epoch % self.conf.epochSaveInterval) == 0 or isBest:
                    self.__saveCheckpoint(epoch, isBest)
        #
        # Copy back the best model.
        self.model.load_state_dict(self.bestModel)
        printColour('Epochs complete!', colours.OKBLUE)
        self.closeLogger(self)
        return self.model

    def train(self):
        ''' Trains a neural netowork according to specified criteria.
        '''
        #
        # Start cuda loaders.
        for loader in self.cudaLoaders:
            self.cudaLoaders[loader].startThread()
        time.sleep(5)
        #
        # Iterate epochs.
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
                q = self.dataQueues[phase]
                inputs = None
                labels = None
                for idx in range(numMini):
                    #
                    # Grab from dataset.
                    try:
                        inputs, labels = q.get(block = True, timeout = 1)
                    except Empty:
                        printError('Missed a batch')
                        continue
                    #
                    # Backward pass.
                    self.optimizer.zero_grad()
                    out = self.model(inputs)
                    _, preds = torch.max(out.data, 1)
                    loss = self.criteria(out, labels)
                    #
                    #  Backwards pass.
                    start = time.time()
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    #
                    #  Stats.
                    # runningLoss += loss.data[0]
                    # runningCorrect += torch.sum(preds == labels.data)
                    pbar.update(1)
                    end = time.time()
                    print('Elapsed %dus'%((end-start)*1e6))
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
                    'data': {'img': inputs, 'labels': labels},
                    'pred' : preds
                }
                self.logEpoch(self, summary)
                #
                # Save model as needed.
                if (epoch % self.conf.epochSaveInterval) == 0 or isBest:
                    self.__saveCheckpoint(epoch, isBest)
        #
        # Copy back the best model.
        self.model.load_state_dict(self.bestModel)
        printColour('Epochs complete!', colours.OKBLUE)
        self.closeLogger(self)
        #
        # For each of the queues call join.
        print('join queues')
        for q in self.dataQueue:
            self.cudaLoaders[q].join()
        #
        # Close data loaders.
        print('join loaders')
        for loader in self.cudaLoaders:
            self.cudaLoaders[loader].stop()
        #
        # Empty Queues.
        print('dataq')
        for q in self.dataQueue:
            for i in range(loader.qsize()):
                self.dataQueue[q].get()
                self.dataQueue[q].task_done()
        for loader in self.cudaLoaders:
            self.cudaLoaders[loader].join()
        return self.model
