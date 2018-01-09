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
                self.layers.append(fc)

    def forward(self, x, labels):
        feat = self.base(x)
        out = []
        outTensor = None

        # dx = labels.data[:, 0]
        # dy = labels.data[:, 1]
        # idx = dy * self.nOutsX + dx
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

    def pUpdate(self, optimizer, criteria, netOut, labels, meta, phase):
        label = labels
        # print(meta['mask'])
        print(label)
        #
        # Translate the relative coordinate to the absolute coordinate.
        #
        # Backward pass.
        optimizer.zero_grad()
        # print(idx)
        # print(netOut)
        # _, preds = torch.max(netOut[idx].data, 1)


        loss = criteria(netOut, label, weight = None)
        #
        #  Backwards pass.
        if phase == 'train':
            loss.backward()
            optimizer.step()
        return preds, loss
