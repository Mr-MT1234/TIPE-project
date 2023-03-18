import numpy as np
from collections import deque
import random
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import RingEnvironment

device = torch.device('cpu')
env = RingEnvironment.RingEnv(radius=120,
                              numVehicles=50,
                              selfDrivingPercentage=0.1,
                              timeStep=1,
                              saveSpeeds=False,
                              selfDrivingAccel=2.,
                              accidentPunishmeent=0,
                              maxIter=3000,
                              renderMode='none')

class DQN(nn.Module):
    def __init__(self, numObs, numAct):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(numObs, 64),
            nn.ReLU(),
            nn.Linear(64,48),
            nn.ReLU(),
            nn.Linear(48,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,numAct)
        )
            
    def forward(self, x):
      out = self.seq(x)
      
      return out
class ReplayMemory:
    def __init__(self,capacity):
        self.queue = deque(maxlen=capacity)
    
    def append(self, experience):
        self.queue.append(experience)
        
    def sample(self, size):
        return random.sample(self.queue, size)
    def __len__(self):
        return  len(self.queue)


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(20000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episodeRewards = []
"""
def plot_rewards(show_result=False):
    plt.figure(1)
    reward_t = torch.tensor(episodeRewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(reward_t.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_t) >= 50:
        means = reward_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
""" 
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = list(zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch[2])), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch[2]
                                                if s is not None])
    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[3])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
num_episodes = 1500
num_models = 10

with open(os.path.join(os.getcwd(),'models/log.txt'), 'w') as sys.stdout:
    for i_model in range(num_models):
        rewards = [0]*num_episodes
        
        t = time.time()
        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state, info = env.reset()
            cummilativeReward = 0

            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated, info, done = env.step(np.array(action))
                cummilativeReward += reward
                rewards[i_episode] = reward
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.append((state, action, next_state, reward))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episodeRewards.append(cummilativeReward)
                    #plot_rewards()
                    sys.stdout.flush()
                    break

        elapsed_time = time.time() - t
        outputDir = os.path.join(os.getcwd(),f'models/model{i_model}/') 
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        print(f'Model {i_model} trained | elapsed time : {time.strftime("%H:%M:%S"),elapsed_time }')
        torch.save(policy_net, os.path.join(outputDir,'model.pt'))
        
        plt.clf()
        plt.plot(rewards)
        avg = [np.mean(rewards[max(i-50,0): i]) for i in range(len(rewards))]
        plt.plot(avg)
        plt.savefig(os.path.join(outputDir,'progress.png'))
        
        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(20000)
        env.reset()

env.close()