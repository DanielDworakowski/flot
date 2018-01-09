import torch
import torch.nn as nn
from torchvision import transforms, models

class GenericModel(nn.Module):

    def __init__(self, model):
        super(GenericModel, self).__init__()
        self.model = model
        #
        # specify the default modification required if the network is a normal
        # resnet18.
        if isinstance(model, type(models.resnet18)):
            self.resizeFC()

    def forward(self, x, y):
        return self.model(x)

    def resizeFC(self):
        numFeat = self.model.fc.in_features
        self.model.fc = nn.Linear(numFeat, 2) # Binary classification.

    def pUpdate(self, optimizer, criteria, netOut, labels, meta, phase):
        #
        # Backward pass.
        optimizer.zero_grad()
        _, preds = torch.max(netOut.data, 1)
        loss = criteria(netOut, labels)
        #
        #  Backwards pass.
        if phase == 'train':
            loss.backward()
            optimizer.step()
        return preds, loss
