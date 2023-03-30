import torch as tr
import numpy as np
from Agent import Agent
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import os, sys
import time

#env = gym.make('Pendulum-v1')
env = gym.make('LunarLanderContinuous-v2')
device = tr.device('cpu')

bounds = (env.action_space.low[0],env.action_space.high[0]) 

agent = Agent (
              device=device,
              stateDim=env.observation_space.shape[0],
              actionDim=env.action_space.shape[0],
              discount=0.99,
              memoryCap=1000000,
              batchSize=64,
              noise=0.1,
              tau=0.005,
              lrActor=0.001,
              lrCritic=0.002,
              layersActor=[512,512],
              layersCritic=[512,512],
              outBounds= bounds,
              savePath=''
              )

EPISODES = 1000
rewards = []
avg = []
running = True
plt.ion()

def train():
    global EPISODES, rewards
    
    log = open(os.path.join(os.getcwd(), 'log.txt'),'w')
    for episode in range(EPISODES):
        begin = time.time()
        if not running:
            break
        totalReward = 0
        obs, info = env.reset()
        obs = tr.tensor(obs)
        Done = False
        
        while not Done and running:
            action = agent.act(obs,True).numpy()
            newObs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs,action,newObs,reward,done)
            
            agent.learn()
            
            obs = tr.tensor(newObs)
            Done = done
            totalReward += reward
            plt.pause(0.001)

        
        now = time.time()
        average = np.mean(rewards[max(-100,-len(rewards)):])
        
        print(f'episode:{episode} \t| reward:{totalReward} \t|took:{now-begin}s \t| avrege:{average}')
        print(f'episode:{episode} \t| reward:{totalReward} \t|took:{now-begin}s \t| avrege:{average}',file=log)
        
        rewards.append(totalReward)
        avg.append(average)
        
        plt.clf()
        plt.plot(rewards)
        plt.plot(avg)
    log.close()

print('------------------------------- Training -------------------------------')
        

train()

print('-------------------------------saving plot------------------ -------------')

plt.plot(rewards)
plt.plot(avg)

savePath = os.path.join(os.getcwd(),'plots')
if not os.path.exists(savePath):
    os.makedirs(savePath)

plt.savefig(savePath)

print('-------------------------------    Done    -------------------------------')

