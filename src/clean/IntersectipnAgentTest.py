import os
import time
from datetime import datetime as dt

import torch as tr
import numpy as np
import matplotlib.pyplot as plt

from Agent import ConvAgent
from IntersectionEnvironment import IntersectionEnv 

envName = 'Intersection - 4 reward - 2'
AUTO_COUNT = 2

env = IntersectionEnv(
    armLength    = 100,
    bodyLength   = 100,
    spawnRate    = 0.75,
    autoCount    = AUTO_COUNT,
    timeStep     = 1/10,
    maxSpeed     = 30,
    turningSpeed = 3,
    maxAccel     = 4,
    maxIter      = 2000
)

device = tr.device('cpu')

currentDate = dt.now()

bounds = (env.action_space.low,env.action_space.high)
actionDim = env.action_space.shape[0]
agentPath = "" #Chemin ou les donnees de l'agent sont enregistrees

agent = ConvAgent(
                device=device,
                stateShape=env.observation_space.shape,
                actionDim=env.action_space.shape[0],
                discount=0.99,
                memoryCap=100000,
                batchSize=64,
                noise=np.array([7,0.5]*AUTO_COUNT),
                tau=0.005,
                lrActor=2e-3,
                lrCritic=3e-3,
                outBounds= bounds,
                learningDecay=0.995,
                noiseDecay=0.995,
                savePath=os.path.join(agentPath,'checkPoint')
            )

agent.loadCheckPoint()

state, _ = env.reset()

stoppingCars = []

for i in range(2500):
    action = agent.act(state, withNoise=False)
    state, _,_,_,_ = env.step(action)
    
    vitesse = 0
    count = 0
    for road,_,_ in env.roads:
        for v in road:
            if v.visualID == 1:
                count += 1
                vitesse += v.velocity
    stoppingCars.append(vitesse/max(1,count))
    env.render()
    
env.close()
env2 = IntersectionEnv(
    armLength    = 100,
    bodyLength   = 100,
    spawnRate    = 0.75,
    autoCount    = 0,
    timeStep     = 1/10,
    maxSpeed     = 30,
    turningSpeed = 3,
    maxAccel     = 4,
    maxIter      = 2500
)

stoppingCars2 = []

for i in range(2500):
    state, _,_,_,_ = env2.step(np.array([]))
    
    vitesse = 0
    count = 0
    for road,_,_ in env2.roads:
        for v in road:
            if v.visualID == 1:
                count += 1
                vitesse += v.velocity
    stoppingCars2.append(vitesse/max(1,count))
    env2.render()
    
env2.close()

T = np.arange(0,250,0.1)
    
plt.plot(T,stoppingCars)
plt.plot(T,stoppingCars2)

plt.legend(['Avec VA', 'Sans VA'])
plt.xlabel('Temps (s)')
plt.ylabel('Vitesse moyenne (m/s)')

plt.show()
    