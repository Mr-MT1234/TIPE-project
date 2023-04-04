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
env = gym.make('LunarLanderContinuous-v2')
device = tr.device('cpu')

bounds = (env.action_space.low,env.action_space.high) 
agentPath = os.path.join(os.getcwd(), f'agents/agent-{time.time()}')
if not os.path.exists(agentPath):
    os.makedirs(agentPath)

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

EPISODES = 2000
INTERACTIVE_MODE = False
rewards = []
avg = []

if INTERACTIVE_MODE:
    plt.ion()
    
# Warm up
WARM_UP_STEPS = 10000
i = 0
while i < WARM_UP_STEPS:
    env1 = gym.make('LunarLanderContinuous-v2')
    state, info = env1.reset()
    done = False
    while not done and i < WARM_UP_STEPS:
        action = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1)])
        newState, reward, terminated, truncated, info = env1.step(action)
        done = terminated or truncated
        agent.remember(state,action,newState,reward, done)
        state = newState
        i += 1
    state, info = env1.reset()


log = open(os.path.join(agentPath, 'log.txt'),'w')
def train():
    global EPISODES, rewards,log
    
    bestAvg = float('-inf')
    for episode in range(EPISODES):
        begin = time.time()

        totalReward = 0
        obs, info = env.reset()
        obs = tr.tensor(obs)
        done = False
        
        last = begin
        while not done:
            action = agent.act(obs,True)
            newObs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs,action,newObs,reward,done)
            
            aLoss, cLoss = agent.learn()
            
            obs = newObs
            totalReward += reward
            now = time.time()
            if INTERACTIVE_MODE and (now - last) > 1/30:
                plt.pause(0.001)
                last = now

        rewards.append(totalReward)
        
        now = time.time()
        average = np.mean(rewards[max(-100,-len(rewards)):])
        if average > bestAvg:
            bestAvg = average
            agent.save()
            
        agent.decay()
        
        print(f'episode:{episode} \t| reward:{totalReward} \t| took:{now-begin}s \t| avrege:{average} \t| loss:(actor:{aLoss},critic:{cLoss})')
        print(f'episode:{episode} \t| reward:{totalReward} \t| took:{now-begin}s \t| avrege:{average} \t| loss:(actor:{aLoss},critic:{cLoss})',file=log)
        
        avg.append(average)
        
        if INTERACTIVE_MODE:
            plt.clf()
            plt.plot(rewards)
            plt.plot(avg)

print('------------------------------- Training -------------------------------')

try:
    train()
except Exception as e:
    if not isinstance(KeyboardInterrupt,e):
        print(f'Error: {e}')
    pass
finally:
    print('-------------------------------saving plot------------------ -------------')

    plt.plot(rewards, color='lightblue', label='Reward')
    plt.plot(avg, color=(85/255,3/255,250/255), label='Average')

    plt.title('Evolution of trainning')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.savefig(os.path.join(agentPath, 'evolution.pdf'), dpi=150)
    log.close()
    print('-------------------------------    Done    -------------------------------')

