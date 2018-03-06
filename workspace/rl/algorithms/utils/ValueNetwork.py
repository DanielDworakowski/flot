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
        self.mini_batch_size = 5
  
    def forward(self, x):
        output = None
        if x.shape[0] > self.mini_batch_size:
            output = []
            idxs = list(range(self.mini_batch_size,x.shape[0],self.mini_batch_size))
            last_idx = 0
            for i in idxs:
                model_out = self.model(x[last_idx:i,:,:,:]).data.cpu().numpy()
                output.append(model_out)
                del model_out
                torch.cuda.empty_cache()
                last_idx = i
            model_out = self.model(x[last_idx:,:,:,:]).data.cpu().numpy()
            output.append(model_out)
            del model_out
            torch.cuda.empty_cache()
            output = np.concatenate(output)
        else:
            output = self.model(x).cpu().numpy()
        return output

    def compute(self, observations):
        observations  = [self.transform(obs) for obs in observations]
        observations = torch.autograd.Variable(torch.stack(observations)).type(self.dtype.FloatTensor) 
        return self.forward(observations)

    def train(self, batch_size, observations_batch, returns_batch, learning_rate):
        returns_batch = np.squeeze(np.array(returns_batch))
        optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        observations_batch  = torch.stack([self.transform(obs) for obs in observations_batch])
        if observations_batch.shape[0] > self.mini_batch_size:
            idxs = list(range(self.mini_batch_size,observations_batch.shape[0],self.mini_batch_size))
            last_idx = 0
            model_outs = []
            targets = []
            for i in idxs:
                obs = torch.autograd.Variable(observations_batch[last_idx:i,:,:,:]).type(self.dtype.FloatTensor)
                model_out = self.model(obs)
                target = torch.autograd.Variable(torch.Tensor(returns_batch[last_idx:i])).type(self.dtype.FloatTensor)
                model_outs.append(model_out)
                targets.append(target)
                del obs
                del model_out
                del target
                torch.cuda.empty_cache()
                last_idx = i
            obs = torch.autograd.Variable(observations_batch[last_idx:,:,:,:]).type(self.dtype.FloatTensor)
            model_out = self.model(obs)            
            model_outs.append(model_out)
            targets.append(target)
            del obs
            del model_out
            del target
            torch.cuda.empty_cache()
            model_outs = torch.cat(model_outs)
            targets = torch.cat(targets)
        else:
            model_outs = self.model(torch.autograd.Variable(observations_batch).type(self.dtype.FloatTensor))
            targets = torch.autograd.Variable(torch.Tensor(returns_batch)).type(self.dtype.FloatTensor)

        optimizer.zero_grad()
        loss = self.loss_fn(model_outs, targets)
        loss.backward()
        optimizer.step()
        
        pdb.set_trace()

        return True
  