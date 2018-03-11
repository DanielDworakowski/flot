import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
from PIL import Image
import itertools

import pdb

class A2CValueNetwork(torch.nn.Module):
    def __init__(self, dtype):
        super(A2CValueNetwork, self).__init__()
        self.dtype = dtype
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224), interpolation=Image.CUBIC), transforms.ToTensor()])
        self.loss_fn = torch.nn.MSELoss()
        self.mini_batch_size = 32
  
    def forward(self, x):
        output = None
        if x.shape[0] > self.mini_batch_size:
            output = []
            idxs = list(range(self.mini_batch_size,x.shape[0],self.mini_batch_size))
            last_idx = 0
            for i in idxs:
                model_out = self.model(torch.autograd.Variable(x[last_idx:i,:,:,:],volatile=True).type(self.dtype.FloatTensor)).data.cpu().numpy()
                output.append(model_out)
                last_idx = i
            model_out = self.model(torch.autograd.Variable(x[last_idx:,:,:,:],volatile=True).type(self.dtype.FloatTensor)).data.cpu().numpy()
            output.append(model_out)
            output = np.concatenate(output)
        else:
            output = self.model(torch.autograd.Variable(x,volatile=True).type(self.dtype.FloatTensor)).cpu().numpy()
        return output

    def compute(self, observations):
        observations  = [self.transform(obs) for obs in observations]
        return self.forward(torch.stack(observations))

    def train(self, batch_size, observations_batch, returns_batch, learning_rate):
        returns_batch = np.squeeze(np.array(returns_batch))
        optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        observations_batch  = torch.stack([self.transform(obs) for obs in observations_batch])
        if observations_batch.shape[0] > self.mini_batch_size:
            idxs = list(range(self.mini_batch_size,observations_batch.shape[0],self.mini_batch_size))
            last_idx = 0
            losses = []
            for i in idxs:
                obs = torch.autograd.Variable(observations_batch[last_idx:i,:,:,:]).type(self.dtype.FloatTensor)
                model_out = self.model(obs)
                target = torch.autograd.Variable(torch.Tensor(returns_batch[last_idx:i])).type(self.dtype.FloatTensor)
                last_idx = i
                optimizer.zero_grad()
                loss = self.loss_fn(model_out, target)
                losses.append(loss.cpu().data.numpy()[0])
                loss.backward()
                optimizer.step()
                last_idx = i
            obs = torch.autograd.Variable(observations_batch[last_idx:,:,:,:]).type(self.dtype.FloatTensor)
            model_out = self.model(obs)
            target = torch.autograd.Variable(torch.Tensor(returns_batch[last_idx:])).type(self.dtype.FloatTensor)
            optimizer.zero_grad()
            loss = self.loss_fn(model_out, target)
            losses.append(loss.cpu().data.numpy()[0])
            value_network_loss = np.mean(losses)
            loss.backward()
            optimizer.step()
        else:
            model_out = self.model(torch.autograd.Variable(observations_batch).type(self.dtype.FloatTensor))
            target = torch.autograd.Variable(torch.Tensor(returns_batch)).type(self.dtype.FloatTensor)
            optimizer.zero_grad()
            loss = self.loss_fn(model_out, target)
            value_network_loss = loss.cpu().data.numpy()[0]
            loss.backward()
            optimizer.step()

        return value_network_loss
  