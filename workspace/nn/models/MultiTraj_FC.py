import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

class Resnet_Multifc(nn.Module):

    def __init__(self, nOuts):
        super(Resnet_Multifc, self).__init__()
        #
        # The base network that is being used for its feature extractor.
        self.base = models.resnet18(pretrained = True)
        #
        # Create multiple FC layers for each of the outputs.
        numFeat = self.base.fc.in_features
        #
        # Remove the last FC in order to create our own list.
        modules = list(self.base.children())[:-1] # delete the last fc layer.
        self.base = nn.Sequential(*modules)
        #
        # FC layers.
        self.fc = []
        self.rangeX = 2 * nOuts[0] + 1
        self.rangeY = 2 * nOuts[1] + 1
        self.layers = nn.ModuleList()
        for i in range(self.rangeY):
            for j in range(self.rangeX):
                fc = nn.Linear(numFeat, 2)
                self.fc.append(fc) # Binary classifier.
                self.add_module('fc%s'%len(self.fc), fc)

    def forward(self, x):
        feat = self.base(x)
        out = []
        outTensor = None
        #
        # Iterate through each of the outputs and get their energies.
        # It is technically not neccessary to do this given that we can obtain
        # label and just forward the corresponding module.
        for row in range(self.rangeY):
            for col in range(self.rangeX):
                idx = row * self.rangeX + col
                out.append(self.fc[idx](feat.squeeze()))
        outTensor = torch.cat(out, dim=1)
        return outTensor

    def getActivations(self, netOut, labels):
        label = labels[:,0]
        mask = labels[:,1:]
        #
        # The dimensions disred for the tensor.
        x = label.size()[0]
        y = 2
        classActivation = torch.masked_select(netOut, mask.byte()).view(x,y)
        return classActivation, label

    def getClassifications(self, netOut, metric):
        #
        # Iterate over the columns and build probabilities.
        nCols = int(netOut.shape[1] / 2)
        out = torch.zeros((netOut.shape[0], nCols))
        for idx in range(nCols):
            activations = netOut[:, 2 * idx : 2 * idx + 2]
            out[:, idx] = metric(activations).data[:, 1] # Get the probaility of the positive class.
        return out

    def pUpdate(self, optimizer, criteria, netOut, labels, meta, phase):
        #
        # Get the activations.
        classActivation, label = self.getActivations(netOut, labels)
        #
        # Backward pass.
        optimizer.zero_grad()
        _, preds = torch.max(classActivation.data, 1)
        loss = criteria(classActivation, label)
        #
        #  Backwards pass.
        if phase == 'train':
            loss.backward()
            optimizer.step()
        dCorrect = torch.sum(preds == label.data)
        return preds, loss, dCorrect
