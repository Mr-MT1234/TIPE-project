import os
import time
from datetime import datetime as dt

import gymnasium as gym
import torch as tr
import numpy as np
import matplotlib.pyplot as plt

from Agent import ConvAgent, Agent
from RingEnvironmentSimple import *
import IntersectionEnvironment 

from collections import deque

import keyboard

SAVING = True
EPISODE_TIME = 150
TIME_STEP = 1/10
EPISODES = 2000
INTERACTIVE_MODE = False
CHECKPOINT_INTERVALLE = 100

envName = 'Intersection - 4'

myParams = IntersectionEnvironment.IDM_CONTROLLER_DEFAULT_PARAMS.copy()
myParams['imperfection'] = 1/5000
myParams['v'] = 17

#env = gym.make('LunarLanderContinuous-v2', render_mode='human')
env = IntersectionEnvironment.IntersectionEnv(
    os.getcwd() + '/SumoIntersectionFiles',
    0.03,
    6,
    timeStep=TIME_STEP,
    IDMControllerParams=myParams,
    maxIter=EPISODE_TIME / TIME_STEP,
    renderMode='human',
)

device = tr.device('cpu')

currentDate = dt.now()

bounds = (env.action_space.low,env.action_space.high)
actionDim = env.action_space.shape[0]
agentPath = os.path.join(os.getcwd(), 'agents/agent {}({}-{}-{} {}.{}.{})'.format(
    envName, currentDate.year,currentDate.month,currentDate.day,currentDate.hour,currentDate.minute, currentDate.second
    ))

agent = ConvAgent(
                device=device,
                stateShape=env.observation_space.shape,
                actionDim=env.action_space.shape[0],
                discount=0.99,
                memoryCap=100000,
                batchSize=64,
                noise=np.array([3,0.2]*6),
                tau=0.005,
                lrActor=2e-3,
                lrCritic=3e-3,
                # layersActor=[256,256],
                # layersCritic=[256,256],
                outBounds= bounds,
                learningDecay=0.995,
                noiseDecay=0.995,
                savePath=os.path.join(agentPath,'checkPoint')
            )
rewards = deque(maxlen=100)
#avg = []

if INTERACTIVE_MODE:
    plt.ion()
    
WARM_UP_STEPS = 1000
i = 0
while i < WARM_UP_STEPS:
    env2 = env
    state, info = env2.reset()
    done = False
    while not done and i < WARM_UP_STEPS:
        action = np.random.uniform(bounds[0],bounds[1])
        newState, reward, terminated, truncated, info = env2.step(action)
        done = terminated or truncated
        agent.remember(state,action,newState,reward, done)
        state = newState
        i += 1
        print(f'Warming up: {i}/{WARM_UP_STEPS}',end='\r')
        
print('Warming up: Done                               ')


if SAVING:
    if not os.path.exists(agentPath):
        os.makedirs(agentPath)
    if not os.path.exists(agent.savePath):
        os.makedirs(agent.savePath)
    log = open(os.path.join(agentPath, 'log.txt'),'w')
    
shouldSave = False
shouldRender = False

def train():
    global EPISODES, rewards,log, shouldSave, shouldRender
    
    bestAvg = float('-inf')
    for episode in range(EPISODES):
        begin = time.time()

        totalReward = 0
        obs, info = env.reset()
        obs = tr.tensor(obs)
        done = False
        
        last = begin
        step = 0
        while not done:
            action = agent.act(obs,True)
            newObs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs,action,newObs,reward,done)
            
            aLoss, cLoss = agent.learn()
            
            obs = newObs
            totalReward += reward
            now = time.time()
            print(f'{step=}', end='\r')
            step += 1
            
            if shouldSave:
                save()       
                shouldSave = False
                
            # if shouldRender and not env.isRenderReady:
            #     env.InitRender()
            # if not shouldRender and env.isRenderReady:
            #     env.EndRender()
                
            if (now - last) > 1/30 and shouldRender:
                env.render()
                last = now
            
            # if INTERACTIVE_MODE and (now - last) > 1/30:
            #     plt.pause(0.001)
            #     last = now

        rewards.append(totalReward)
        
        now = time.time()
        average = np.mean(rewards)
        # if SAVING and average > bestAvg:
            # path = os.path.join(agentPath, 'BestAverage')
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # bestAvg = average
            # agent.save(path)
            
        #agent.decay()
        
        print(f'episode:{episode} \t| reward:{totalReward:.2f} \t| took:{now-begin:.2f}s \t| avrege:{average:.2f})')
        if SAVING:
            print(f'episode:{episode} \t| reward:{totalReward:.2f} \t| took:{now-begin:.2f}s \t| avrege:{average:.2f})',file=log)
            
        if episode % CHECKPOINT_INTERVALLE == 0 and SAVING:
            agent.saveCheckPoint()
        #avg.append(average)
        
        # if INTERACTIVE_MODE:
        #     plt.clf()
        #     plt.plot(rewards)
        #     plt.plot(avg)

SnapshotID = 0

def save():
    global SnapshotID 
    print('-------------------------------saving-------------------------------')
    if not os.path.exists(agentPath):
        os.makedirs(agentPath)
    
    agentSavePath = os.path.join(agentPath, f'snapshot-{SnapshotID}')
    os.makedirs(agentSavePath)
    
    agent.save(agentSavePath)
    SnapshotID += 1
    
    # plt.clf()
    # plt.plot(rewards, color='lightblue', label='Reward')
    # plt.plot(avg, color=(85/255,3/255,250/255), label='Average')

    # plt.title('Evolution of trainning')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.legend()

    # plt.savefig(os.path.join(agentPath, 'evolution.pdf'), dpi=150)

def toggleRender():
    global shouldRender
    shouldRender = not shouldRender
def registerSaveCommand():
    global shouldSave
    shouldSave = True

keyboard.add_hotkey('F1', callback=registerSaveCommand)
keyboard.add_hotkey('F2', callback=toggleRender)

print('------------------------------- Training -------------------------------')

try:
    train()
except KeyboardInterrupt as e:
    pass
finally:
    env.close()
    if SAVING:
        save()
        log.close()
    print('-------------------------------    Done    -------------------------------')