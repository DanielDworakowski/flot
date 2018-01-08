import torch
import torch.nn as nn
from torchvision import transforms, models

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
        self.nOutsX, self.nOutsY = nOuts
        self.layers = nn.ModuleList()
        for i in range(self.nOutsY):
            for j in range(self.nOutsX):
                fc = nn.Linear(numFeat, 2)
                self.fc.append(fc) # Binary classifier.
                self.layers.append(fc)

    def forward(self, x):
        feat = self.base(x)
        out = []
        #
        # Iterate through each of the outputs and get their energies.
        # It is technically not neccessary to do this given that we can obtain
        # label and just forward the corresponding module.
        print(self.fc)
        for row in range(self.nOutsY):
            for col in range(self.nOutsX):
                print(feat)
                idx = row * self.nOutsX + col
                print(self.fc[idx])
                out.append(self.fc[idx](feat))
        return out

    def pUpdate(self, optimizer, criteria, netOut, labels, meta, phase):
        dx = labels.data[0][0]
        dy = labels.data[0][1]
        nOutsX, nOutsY = meta['nSteps']
        label = labels.data[0][2]
        #
        # TODO: clean this up so that the meta is not passed back just to get
        # the number of outputs. Maybe wrap a generic model class?
        idx = dy * nOutsX[0] + dx
        #
        # Backward pass.
        optimizer.zero_grad()
        print(netOut)
        _, preds = torch.max(netOut[idx].data, 1)
        loss = criteria(netOut[idx], label)
        #
        #  Backwards pass.
        if phase == 'train':
            loss.backward()
            optimizer.step()
        return preds, loss
