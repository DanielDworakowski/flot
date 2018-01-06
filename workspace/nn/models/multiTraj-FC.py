import torch.nn as nn
from torchvision import transforms, models

class resnet_multifc(nn.module):

    def __init__(self, nOuts):
        super(resnet_multifc, self).__init__()
        #
        # The base network that is being used for its feature extractor.
        self.base = models.resnet18(pretrained = True)
        #
        # Create multiple FC layers for each of the outputs.
        numFeat = net.fc.in_features
        #
        # Remove the last FC in order to create our own list.
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.base = nn.Sequential(*modules)
        #
        # FC layers.
        self.fc = []
        self.nOuts = nOuts
        for i in range(nOuts):
            self.fc.append(nn.Linear(numFeat, len(param.trainSignals) + 1))

    def forward(self, x):
        feat = self.base(x)
        out = []
        #
        # Iterate through each of the outputs.
        for i in range(self.nOuts):
            out.append(self.fc[i](feat))
        return out
