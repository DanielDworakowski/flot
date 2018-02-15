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
        if isinstance(self.model, models.resnet.ResNet):
            self.resizeFC()

    def forward(self, x):
        return self.model(x)

    def resizeFC(self):
        numFeat = self.model.fc.in_features
        self.model.fc = nn.Linear(numFeat, 2) # Binary classification.

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
        # Backward pass.
        optimizer.zero_grad()
        _, preds = torch.max(netOut.data, 1)
        loss = criteria(netOut, labels)
        #
        #  Backwards pass.
        if phase == 'train':
            loss.backward()
            optimizer.step()
        dCorrect = torch.sum(preds == labels.data)
        return preds, loss, dCorrect
