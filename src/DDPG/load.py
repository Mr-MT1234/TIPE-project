import torch as tr
import numpy as np
from Agent import Agent
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import os, sys
import time

#env = gym.make('Pendulum-v1', render_mode='human')
env = gym.make('LunarLanderContinuous-v2',render_mode='human')
device = tr.device('cpu')

agentPath = 'C:/Users/Tribik/Documents/Mohamed/Etudes/MP/TIPE/Code/Ring/src/agents/bestsofar'
bounds = (env.action_space.low,env.action_space.high) 

agent = Agent(
                device=device,
                stateDim=env.observation_space.shape[0],
                actionDim=env.action_space.shape[0],
                discount=0.99,
                memoryCap=1000000,
                batchSize=64,
                noise=0.1,
                tau=0.005,
                lrActor=2e-4,
                lrCritic=3e-4,
                layersActor=[256,256],
                layersCritic=[256,256],
                outBounds= bounds,
                learningDecay=0.995,
                noiseDecay=0.995,
                savePath=agentPath
            )

agent.load()

for episode in range(10000):
    begin = time.time()

    totalReward = 0
    obs, info = env.reset()
    obs = tr.tensor(obs)
    done = False
    
    last = begin
    while not done:
        action = agent.act(obs)
        newObs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
                
        obs = newObs
        totalReward += reward
        now = time.time()
    