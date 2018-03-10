import numpy as np 
import torch
import importlib
from algorithms.utils.utils import *
from algorithms.utils.PolicyNetwork import A2CPolicyNetwork
from algorithms.utils.ValueNetwork import A2CValueNetwork
from tensorboardX import SummaryWriter
import itertools

import pdb

class Agent:
    """
    Advantage Actor Suggestor Algorithm

    """
    def __init__(self,
                 env,
                 training_params = {'min_batch_size':1000,
                                   'total_timesteps':1000000,
                                   'desired_kl':2e-3},
                 algorithm_params = {'gamma':0.99, 
                                    'learning_rate':1e-6}):

        self.env = env
        self.dtype = torch.cuda

        # Getting the shape of observation space and action space of the environment
        self.observation_shape = self.env.observation_shape
        self.action_shape = self.env.action_shape
        
        # Hyper Parameters
        self.training_params = training_params
        self.algorithm_params = algorithm_params
        
        ##### Networks #####
        self.value_network = A2CValueNetwork(self.dtype)
        self.policy_network = A2CPolicyNetwork(self.dtype, self.action_shape[0])

        self.value_network.cuda()
        self.policy_network.cuda()

        ##### Logging #####
        self.writer = SummaryWriter()
        self.save()

    def save(self):
        torch.save(self.value_network.state_dict(), "value_network.pt")
        torch.save(self.policy_network.state_dict(), "policy_network.pt")

    # Collecting experience (data) and training the agent (networks)
    def train(self):

        # Keeping count of total timesteps and episodes of environment experience for stats
        total_timesteps = 0
        total_episodes = 0

        # KL divergence, used to adjust the learning rate
        kl = 0

        # Keeping track of the best averge reward
        best_average_reward = -np.inf

        ##### Training #####

        # Training iterations
        while total_timesteps < self.training_params['total_timesteps']:

            if total_episodes == 1:
                pdb.set_trace()

            # Collect batch of data
            trajectories, returns, undiscounted_returns, advantages, batch_size, episodes = self.collect_trajs(total_timesteps)
            observations_batch, actions_batch, rewards_batch, returns_batch, next_observations_batch, advantages_batch = self.traj_to_batch(trajectories, returns, advantages) 

            # update total timesteps and total episodes
            total_timesteps += batch_size
            total_episodes += episodes

            # Learning rate adaptation
            learning_rate = self.algorithm_params['learning_rate']

            # Average undiscounted return for the last data collection
            average_reward = np.mean(undiscounted_returns)

            if average_reward > best_average_reward:
                self.save()
                best_average_reward = average_reward

            ##### Optimization #####

            value_network_loss = self.train_value_network(batch_size, observations_batch, returns_batch, learning_rate)
            self.writer.add_scalar("data/value_network_loss", value_network_loss, total_timesteps)
            torch.cuda.empty_cache()

            policy_network_loss = self.train_policy_network(observations_batch, actions_batch, advantages_batch, learning_rate)
            self.writer.add_scalar("data/policy_network_loss", policy_network_loss, total_timesteps)
            torch.cuda.empty_cache()

            self.print_stats(total_timesteps, total_episodes, best_average_reward, average_reward, policy_network_loss, value_network_loss, learning_rate, batch_size)

        self.writer.close()

    ##### Helper Functions #####

    # Collect trajectores
    def collect_trajs(self, total_timesteps):
        # Batch size and episodes experienced in current iteration
        batch_size = 0
        episodes = 0

        # Lists to collect data
        trajectories, returns, undiscounted_returns, advantages = [], [], [], []

        ##### Collect Batch #####

        # Collecting minium batch size or minimum episodes of experience
        while batch_size < self.training_params['min_batch_size']:
                          
            ##### Episode #####

            # Run one episode
            observations, actions, rewards, dones = self.run_one_episode()

            ##### Data Appending #####

            # Get sum of reward for this episode
            undiscounted_returns.append(np.sum(rewards))

            # Update the counters
            batch_size += len(rewards)
            total_timesteps += len(rewards)
            episodes += 1

            # Episode trajectory
            trajectory = {"observations":np.array(observations), "actions":np.array(actions), "rewards":np.array(rewards), "dones":np.array(dones)}
            trajectories.append(trajectory)

            # Computing the discounted return for this episode (NOT A SINGLE NUMBER, FOR EACH OBSERVATION)
            return_ = discount(trajectory["rewards"], self.algorithm_params['gamma'])

            # Compute the value estimates for the observations seen during this episode
            values = np.squeeze(self.value_network.compute(observations))

            # Computing the advantage estimate
            advantage = return_ - values
            returns.append(return_)
            advantages.append(advantage)

        return [trajectories, returns, undiscounted_returns, advantages, batch_size, episodes]

    # Run one episode
    def run_one_episode(self):

        # Restart env
        observation = self.env.reset()

        # Flag that env is in terminal state
        done = False

        observations, actions, rewards, dones = [], [], [], []
        while not done:
            # Collect the observation
            observations.append(observation)

            # Sample action with current policy
            action = self.compute_action(observation)
            # Take action in environment
            observation, reward, done = self.env.step(action)

            # Collect reward and action
            rewards.append(reward)
            actions.append(action)
            dones.append(done)

        return [observations, actions, rewards, dones]

    # Compute action using policy network
    def compute_action(self, observation):
        action = self.policy_network.compute(observation)
        return action

    # Convert trajectories to batches
    def traj_to_batch(self, trajectories, returns, advantages):
        ##### Data Prep #####
          
        # Observations for this batch
        observations_batch = np.concatenate([trajectory["observations"] for trajectory in trajectories])
        next_observations_batch = np.roll(observations_batch, 1, axis=0)
        next_observations_batch[0,:] = observations_batch[0,:]

        # Actions for this batch, reshapeing to handel 1D action space
        actions_batch = np.concatenate([trajectory["actions"] for trajectory in trajectories]).reshape([-1,self.action_shape[0]])

        # Rewards of the trajectory as a batch
        rewards_batch = np.concatenate([trajectory["rewards"] for trajectory in trajectories]).reshape([-1,1])

        # Binary dones from environment in a batch
        dones_batch = np.concatenate([trajectory["dones"] for trajectory in trajectories])

        # Discounted returns for this batch. itertool used to make batch into long np array
        returns_batch = np.array(list(itertools.chain.from_iterable(returns))).reshape([-1,1])

        # Advantages for this batch. itertool used to make batch into long np array
        advantages_batch = np.array(list(itertools.chain.from_iterable(advantages))).flatten().reshape([-1,1])

        return [observations_batch, actions_batch, rewards_batch, returns_batch, next_observations_batch, advantages_batch]

    # Train value network
    def train_value_network(self, batch_size, observations_batch, returns_batch, learning_rate):
        loss = self.value_network.train(batch_size, observations_batch, returns_batch, learning_rate)
        return loss

    # Train policy network
    def train_policy_network(self, observations_batch, actions_batch, advantages_batch, learning_rate):
        loss = self.policy_network.train(observations_batch, actions_batch, advantages_batch, learning_rate)
        return loss

    # Print stats
    def print_stats(self, total_timesteps, total_episodes, best_average_reward, average_reward, policy_network_loss, value_network_loss, learning_rate, batch_size):
        ##### Reporting Performance #####
          
        # Printing performance progress and other useful infromation
        print("_______________________________________________________________________________________________________________________________________________________________________________________________________________")
        print("{:>15} {:>15} {:>15} {:>15} {:>20} {:>20} {:>10} {:>15}".format("total_timesteps", "episodes", "best_reward", "reward", "policy_loss", "value_loss", "lr", "batch_size"))
        print("{:>15} {:>15} {:>15.2f} {:>15.2f} {:>20.2f} {:>20.2f} {:>10.2E} {:>15}".format(total_timesteps, total_episodes, best_average_reward, average_reward, policy_network_loss, value_network_loss, learning_rate, batch_size))

