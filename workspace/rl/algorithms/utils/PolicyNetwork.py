import numpy as np 
import torch
import importlib
from algorithms.utils.utils import *
import itertools

class A2CPolicy(torch.nn.Module):
  def __init__(self):
      super(A2CPolicy, self).__init__()
      self.policy_network_class = importlib.import_module("architectures." + self.network_params['network_type'])
      self.current_policy_network = self.policy_network_class.Network(self.sess, self.observations, policy_output_shape, "current_policy_network", self.network_params['network_size'])
  
  # In a PyTorch model, you only have to define the forward pass. PyTorch computes the backwards pass for you!
  def forward(self, x):
      x = self.linear1(x)
      x = F.relu(x)
      x = self.linear2(x)
      x = F.relu(x)
      x = self.linear3(x)
      x = F.relu(x) 
      return x
  
  # Only the Actor head
  def get_action_probs(self, x):
      x = self(x)
      action_probs = F.softmax(self.actor(x))
      return action_probs
  
  # Only the Critic head
  def get_state_value(self, x):
      x = self(x)
      state_value = self.critic(x)
      return state_value
  
  # Both heads
  def evaluate_actions(self, x):
      x = self(x)
      action_probs = F.softmax(self.actor(x))
      state_values = self.critic(x)
      return action_probs, state_values

