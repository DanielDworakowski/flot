import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
from PIL import Image
import itertools

import pdb

class A2CValueNetwork(torch.nn.Module):
    def __init__(self, dtype, obs_dim):
        super(A2CValueNetwork, self).__init__()
        self.dtype = dtype

        self.batchnorm0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 15, 8, stride=4)
        self.pool1 = torch.nn.AvgPool2d(8,4)
        self.batchnorm1 = torch.nn.BatchNorm2d(15)
        self.conv2 = torch.nn.Conv2d(15, 30, 4, stride=2)
        self.pool2 = torch.nn.AvgPool2d(4,2)
        self.batchnorm2 = torch.nn.BatchNorm2d(30)
        self.conv3 = torch.nn.Conv2d(30, 30, 4, stride=2)
        self.pool3 = torch.nn.AvgPool2d(4,2)
        self.batchnorm3 = torch.nn.BatchNorm2d(30)
        self.fc1 = torch.nn.Linear(1470, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 1)

        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((150,150), interpolation=Image.CUBIC), transforms.ToTensor()])
        self.loss_fn = torch.torch.nn.MSELoss()
        self.mini_batch_size = 128

    def model(self, x):
        x = self.batchnorm0(x)
        x = torch.nn.functional.relu(self.batchnorm1( self.conv1(x) + torch.cat([self.pool1(x)]*5,1) ))
        x = torch.nn.functional.relu(self.batchnorm2( self.conv2(x) + torch.cat([self.pool2(x)]*2,1) ))
        x = torch.nn.functional.relu(self.batchnorm3( self.conv3(x) + torch.cat([self.pool3(x)]*1,1) ))
        x = x.view(-1, int(1470))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x
  
    def forward(self, x):
        output = None
        if x.shape[0] > self.mini_batch_size:
            output = []
            idxs = list(range(self.mini_batch_size,x.shape[0],self.mini_batch_size))
            last_idx = 0
            for i in idxs:
                obs = torch.autograd.Variable(x[last_idx:i,:,:,:]).type(self.dtype.FloatTensor)
                model_out = self.model(obs).data.cpu().numpy()
                output.append(model_out)
                last_idx = i
            obs = torch.autograd.Variable(x[last_idx:,:,:,:]).type(self.dtype.FloatTensor)
            model_out = self.model(obs).data.cpu().numpy()
            output.append(model_out)
            output = np.concatenate(output)
        else:
            output = self.model(torch.autograd.Variable(x,volatile=True).type(self.dtype.FloatTensor)).data.cpu().numpy()
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
            obs = torch.autograd.Variable(observations_batch).type(self.dtype.FloatTensor)
            model_out = self.model(obs)
            target = torch.autograd.Variable(torch.Tensor(returns_batch)).type(self.dtype.FloatTensor)
            optimizer.zero_grad()
            loss = self.loss_fn(model_out, target)
            value_network_loss = loss.cpu().data.numpy()[0]
            loss.backward()
            optimizer.step()

        return value_network_loss
  