import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
from PIL import Image
import itertools

import pdb

class A2CPolicyNetwork(torch.nn.Module):
    def __init__(self, dtype, action_dim):
        super(A2CPolicyNetwork, self).__init__()
        self.dtype = dtype
        self.action_dim = action_dim
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.action_dim*2)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224), interpolation=Image.CUBIC), transforms.ToTensor()])
  
    def forward(self, x):
        return self.model(x)

    def compute(self, observation):
        observation = torch.autograd.Variable(self.transform(observation)).type(self.dtype.FloatTensor).unsqueeze(0)
        # plt.imshow(observation.data.cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')        
        model_out = self.forward(observation).squeeze()
        mean, std_dev = model_out[:self.action_dim].data, torch.exp(model_out[self.action_dim:].data)
        distribution = torch.distributions.Normal(mean, std_dev)
        return distribution.sample().cpu().numpy()
  
    def train(self, observations_batch, actions_batch, advantages_batch, learning_rate):
        pdb.set_trace()
        advantages_batch = np.squeeze(np.array(advantages_batch))
        actions_batch = np.squeeze(np.array(actions_batch))
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
  
