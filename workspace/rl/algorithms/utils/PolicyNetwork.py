import numpy as np 
import torch
from torchvision import transforms, models
import importlib
from algorithms.utils.utils import *
from PIL import Image
import itertools

import pdb

class A2CPolicyNetwork(torch.nn.Module):
    def __init__(self, dtype, action_dim, obs_dim, aux_dim):
        super(A2CPolicyNetwork, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.batchnorm0 = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 24, 8, stride=4)
        self.pool1 = torch.nn.AvgPool2d(8,4)
        self.batchnorm1 = torch.nn.BatchNorm2d(24)
        self.conv2 = torch.nn.Conv2d(24, 48, 4, stride=2)
        self.pool2 = torch.nn.AvgPool2d(4,2)
        self.batchnorm2 = torch.nn.BatchNorm2d(48)
        self.conv3 = torch.nn.Conv2d(48, 48, 4, stride=2)
        self.pool3 = torch.nn.AvgPool2d(4,2)
        self.batchnorm3 = torch.nn.BatchNorm2d(48)
        self.conv4 = torch.nn.Conv2d(48, 48, 4, stride=2)
        self.pool4 = torch.nn.AvgPool2d(4,2)
        self.batchnorm4 = torch.nn.BatchNorm2d(48)
        self.lstm1 = torch.nn.LSTM(192, 128, 1)
        self.fc1 = torch.nn.Linear(192, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, self.action_dim*2+aux_dim)
        
        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.uniform(self.fc3.weight, -3e-4, 3e-4)

        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((64,64), interpolation=Image.CUBIC), transforms.Grayscale(1), transforms.ToTensor()])
        self.loss_fn_aux = torch.torch.nn.MSELoss()       

    def model(self, x):
        x = self.batchnorm0(x)
        x = torch.nn.functional.relu(self.batchnorm1( self.conv1(x) + torch.cat([self.pool1(x)]*24,1) ))
        x = torch.nn.functional.relu(self.batchnorm2( self.conv2(x) + torch.cat([self.pool2(x)]*2,1) ))
        x = torch.nn.functional.relu(self.batchnorm3( self.conv3(x) + torch.cat([self.pool3(x)]*1,1) ))
        # x = torch.nn.functional.relu(self.batchnorm4( self.conv4(x) + torch.cat([self.pool4(x)]*1,1) ))
        x = x.view(-1, int(192))
        x = x.unsqueeze(1)
        x = self.lstm1(x)[0]
        x = x.view(-1, int(128))
        # x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x
  
    def forward(self, x):
        return self.model(x)

    def multi_frame(self, obs_batch, num_frame=3):
        new_obs_batch = []
        for i in range(len(obs_batch)):
            new_obs_batch.append(torch.cat(list(reversed(obs_batch[max(i+1-num_frame,0):i+1])) + [obs_batch[0]]*max(0,num_frame-i-1)))
        return new_obs_batch

    def compute(self, obs_batch):

        # num_frame = 3
        # i = len(obs_batch)-1

        # stacked_obs = list(reversed(obs_batch[max(i+1-num_frame,0):i+1])) + [obs_batch[0]]*max(0,num_frame-i-1)

        observation  = self.transform(obs_batch[-1])

        observation = torch.autograd.Variable(observation,volatile=True).type(torch.FloatTensor).unsqueeze(0)

        # plt.imshow(observation.data.cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
        model_out = self.forward(observation).squeeze()
        mean, std_dev = model_out[:self.action_dim].data, torch.log(torch.exp(model_out[self.action_dim:2*self.action_dim])+1).data
        # std_dev = std_dev*0 + 0.3

        distribution = torch.distributions.Normal(mean, std_dev)

        return distribution.sample().cpu().numpy()
  
    def train(self, observations_batch, actions_batch, advantages_batch, learning_rate, auxs_batch):
        optimizer = torch.optim.Adam(self.parameters(), learning_rate)

        advantages_batch = np.squeeze(np.array(advantages_batch))
        actions_batch = np.array(actions_batch)
        auxs_batch  = np.array(auxs_batch)
        observations_batch  = [self.transform(obs) for obs in observations_batch]
        # observations_batch = self.multi_frame(observations_batch)
        observations_batch = torch.stack(observations_batch)

        # rand_idx = np.random.permutation(observations_batch.shape[0])
        # advantages_batch = advantages_batch[rand_idx]
        # actions_batch = actions_batch[rand_idx]
        # observations_batch  = observations_batch[rand_idx,:,:,:]
        # auxs_batch  = auxs_batch[rand_idx,:]

        obs = torch.autograd.Variable(observations_batch).type(torch.FloatTensor)
        action = torch.autograd.Variable(torch.Tensor(actions_batch)).type(torch.FloatTensor) 
        advantage = torch.autograd.Variable(torch.Tensor(advantages_batch)).type(torch.FloatTensor) 
        model_out = self.model(obs)
        mean, std_dev = model_out[:,:self.action_dim], torch.log(torch.exp(model_out[:,self.action_dim:2*self.action_dim])+1)
        # std_dev = std_dev*0 + 0.3
        distribution = torch.distributions.Normal(mean, std_dev)
        optimizer.zero_grad()
        aux_target = torch.autograd.Variable(torch.Tensor(auxs_batch)).type(torch.FloatTensor)
        loss = torch.mean(-distribution.log_prob(action)*advantage.unsqueeze(1)) + 10*self.loss_fn_aux(model_out[:,2*self.action_dim:], aux_target)
        loss.backward()
        optimizer.step()
        policy_network_loss = loss.cpu().data.numpy()[0]

        return policy_network_loss
  
