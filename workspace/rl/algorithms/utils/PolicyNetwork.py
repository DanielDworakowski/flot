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
        self.mini_batch_size = 32
  
    def forward(self, x):
        return self.model(x)

    def compute(self, observation):
        observation = torch.autograd.Variable(self.transform(observation),volatile=True).type(self.dtype.FloatTensor).unsqueeze(0)
        # plt.imshow(observation.data.cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')        
        model_out = self.forward(observation).squeeze()
        mean, std_dev = model_out[:self.action_dim].data, torch.exp(model_out[self.action_dim:].data)
        distribution = torch.distributions.Normal(mean, std_dev)
        return distribution.sample().cpu().numpy()
  
    def train(self, observations_batch, actions_batch, advantages_batch, learning_rate):
        optimizer = torch.optim.Adam(self.parameters(), learning_rate)

        advantages_batch = np.squeeze(np.array(advantages_batch))
        actions_batch = np.array(actions_batch)
        observations_batch  = torch.stack([self.transform(obs) for obs in observations_batch])
        rand_idx = np.random.permutation(observations_batch.shape[0])
    
        advantages_batch = advantages_batch[rand_idx]
        actions_batch = actions_batch[rand_idx]
        observations_batch  = observations_batch[rand_idx,:,:,:]
        

        if observations_batch.shape[0] > self.mini_batch_size:
            idxs = list(range(self.mini_batch_size,observations_batch.shape[0],self.mini_batch_size))
            last_idx = 0
            losses = []
            for i in idxs:
                obs = torch.autograd.Variable(observations_batch[last_idx:i,:,:,:]).type(self.dtype.FloatTensor)
                action = torch.autograd.Variable(torch.Tensor(actions_batch[last_idx:i])).type(self.dtype.FloatTensor) 
                advantage = torch.autograd.Variable(torch.Tensor(advantages_batch[last_idx:i])).type(self.dtype.FloatTensor) 
                model_out = self.model(obs)
                mean, std_dev = model_out[:,:self.action_dim], torch.exp(model_out[:,self.action_dim:])
                distribution = torch.distributions.Normal(mean, std_dev)
                optimizer.zero_grad()
                loss = torch.mean(-distribution.log_prob(action)*advantage.unsqueeze(1))
                losses.append(loss.cpu().data.numpy()[0])
                loss.backward()
                optimizer.step()
                last_idx = i
            obs = torch.autograd.Variable(observations_batch[last_idx:,:,:,:]).type(self.dtype.FloatTensor)
            action = torch.autograd.Variable(torch.Tensor(actions_batch[last_idx:])).type(self.dtype.FloatTensor) 
            advantage = torch.autograd.Variable(torch.Tensor(advantages_batch[last_idx:])).type(self.dtype.FloatTensor) 
            model_out = self.model(obs)
            mean, std_dev = model_out[:,:self.action_dim], torch.exp(model_out[:,self.action_dim:])
            distribution = torch.distributions.Normal(mean, std_dev)
            optimizer.zero_grad()
            loss = torch.mean(-distribution.log_prob(action)*advantage.unsqueeze(1))
            losses.append(loss.cpu().data.numpy()[0])
            loss.backward()
            optimizer.step()
            policy_network_loss = np.mean(losses)
        else:
            obs = torch.autograd.Variable(observations_batch).type(self.dtype.FloatTensor)
            action = torch.autograd.Variable(torch.Tensor(actions_batch)).type(self.dtype.FloatTensor) 
            advantage = torch.autograd.Variable(torch.Tensor(advantages_batch)).type(self.dtype.FloatTensor) 
            model_out = self.model(obs)
            mean, std_dev = model_out[:,:self.action_dim], torch.exp(model_out[:,self.action_dim:])
            distribution = torch.distributions.Normal(mean, std_dev)
            optimizer.zero_grad()
            loss = torch.mean(-distribution.log_prob(action)*advantage.unsqueeze(1))
            losses.append(loss.cpu().data.numpy()[0])
            loss.backward()
            optimizer.step()
            policy_network_loss = np.mean(losses)

        return policy_network_loss
  
