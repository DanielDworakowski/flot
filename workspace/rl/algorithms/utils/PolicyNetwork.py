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

        self.batchnorm0 = torch.nn.BatchNorm2d(6)
        self.conv1 = torch.nn.Conv2d(6, 30, 8, stride=4)
        self.pool1 = torch.nn.AvgPool2d(8,4)
        self.batchnorm1 = torch.nn.BatchNorm2d(30)
        self.conv2 = torch.nn.Conv2d(30, 60, 4, stride=2)
        self.pool2 = torch.nn.AvgPool2d(4,2)
        self.batchnorm2 = torch.nn.BatchNorm2d(60)
        self.conv3 = torch.nn.Conv2d(60, 60, 3, stride=1)
        self.pool3 = torch.nn.AvgPool2d(3,1)
        self.batchnorm3 = torch.nn.BatchNorm2d(60)
        self.fc1 = torch.nn.Linear(34560, 512)
        self.fc2 = torch.nn.Linear(512, self.action_dim*2)


        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224), interpolation=Image.CUBIC), transforms.ToTensor()])       
        self.mini_batch_size = 32

    def model(self, x):
        x = self.batchnorm0(x)
        x = torch.nn.functional.relu(self.batchnorm1( self.conv1(x) + torch.cat([self.pool1(x)]*5,1) ))
        x = torch.nn.functional.relu(self.batchnorm2( self.conv2(x) + torch.cat([self.pool2(x)]*2,1) ))
        x = torch.nn.functional.relu(self.batchnorm3( self.conv3(x) + torch.cat([self.pool3(x)]*1,1) ))
        x = x.view(-1, int(34560))
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x
  
    def forward(self, x):
        return self.model(x)

    def compute(self, observations):
        if len(observations) == 1:
            observation_prev = self.transform(observations[-1])
            observation_now = self.transform(observations[-1])
        else:
            observation_prev = self.transform(observations[-2])
            observation_now = self.transform(observations[-1])

        observation = torch.autograd.Variable(torch.cat([observation_now, observation_prev],0) ,volatile=True).type(self.dtype.FloatTensor).unsqueeze(0)

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
        # rand_idx = np.random.permutation(observations_batch.shape[0])
    
        # advantages_batch = advantages_batch[rand_idx]
        # actions_batch = actions_batch[rand_idx]
        # observations_batch  = observations_batch[rand_idx,:,:,:]
        

        if observations_batch.shape[0] > self.mini_batch_size:
            idxs = list(range(self.mini_batch_size,observations_batch.shape[0],self.mini_batch_size))
            last_idx = 0
            losses = []
            for i in idxs:
                obs_cat = torch.cat([observations_batch[last_idx+1:i+1,:,:,:],observations_batch[last_idx:i,:,:,:]],1)
                obs = torch.autograd.Variable(obs_cat).type(self.dtype.FloatTensor)
                action = torch.autograd.Variable(torch.Tensor(actions_batch[last_idx:i])).type(self.dtype.FloatTensor) 
                advantage = torch.autograd.Variable(torch.Tensor(advantages_batch[last_idx:i])).type(self.dtype.FloatTensor) 
                model_out = self.model(obs)
                mean, std_dev = model_out[:,:self.action_dim], torch.exp(model_out[:,self.action_dim:])
                distribution = torch.distributions.Normal(mean, std_dev)
                optimizer.zero_grad()
                loss = torch.mean(-distribution.log_prob(action)*advantage.unsqueeze(1))
                losses.append(loss.cpu().data.numpy()[0])
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.parameters(), 40)
                optimizer.step()
                last_idx = i
            obs_cat = torch.cat([observations_batch[last_idx+1:,:,:,:],observations_batch[last_idx:-1,:,:,:]],1)
            obs = torch.autograd.Variable(obs_cat).type(self.dtype.FloatTensor)
            action = torch.autograd.Variable(torch.Tensor(actions_batch[last_idx:])).type(self.dtype.FloatTensor) 
            advantage = torch.autograd.Variable(torch.Tensor(advantages_batch[last_idx:])).type(self.dtype.FloatTensor) 
            model_out = self.model(obs)
            mean, std_dev = model_out[:,:self.action_dim], torch.exp(model_out[:,self.action_dim:])
            distribution = torch.distributions.Normal(mean, std_dev)
            optimizer.zero_grad()
            loss = torch.mean(-distribution.log_prob(action)*advantage.unsqueeze(1))
            losses.append(loss.cpu().data.numpy()[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), 40)
            optimizer.step()
            policy_network_loss = np.mean(losses)
        else:
            print("minibatchsize")

        return policy_network_loss
  
