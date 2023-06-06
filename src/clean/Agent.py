import os

import torch as tr
import torch.nn as trn
import torch.nn.functional as trf
import torch.optim as opt
import numpy as np

from Networks import *
from Memory import ReplayMemory

class OUNoise:
    def __init__(self, mu, sigma=.15, theta=.8, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros_like(self.mu)
        self.reset()

    def __call__(self):
        x = self.x_prev - self.theta * (self.x_prev - self.mu) * self.dt + self.sigma * np.sqrt(2*self.theta*self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0

class ConvAgent:
    def __init__(self, device, stateShape, actionDim, discount, tau, memoryCap, batchSize,
                    noise, lrActor, lrCritic, outBounds, learningDecay, noiseDecay,savePath):
        self.stateShape = stateShape
        self.actionDim = actionDim
        self.lrActor = lrActor
        self.lrCritic = lrCritic
        self.tau = tau
        self.discount = discount
        self.noise = OUNoise(np.zeros(actionDim),sigma=noise)
        self.savePath = savePath
        self.batchSize = batchSize
        self.outBounds = tr.tensor(outBounds)
        self.device = device
        self.learningDecay = learningDecay
        self.noiseDecay = noiseDecay
        
        actorPath  = os.path.join(savePath, 'actorNet.pt')
        criticPath = os.path.join(savePath, 'criticNet.pt')
        actorTargetPath  = os.path.join(savePath, 'actorTargetNet.pt')
        criticTargetPath = os.path.join(savePath, 'criticTargetNet.pt')
        
        self.memory = ReplayMemory (memoryCap, stateShape, actionDim, batchSize)
        self.actor  = ActorConvNetwork (device, stateShape, actionDim, outBounds, actorPath)
        self.critic = CriticConvNetwork(device, stateShape, actionDim, criticPath) 
        
        self.actorTarget  = ActorConvNetwork (device, stateShape, actionDim, outBounds, actorTargetPath)
        self.criticTarget = CriticConvNetwork(device, stateShape, actionDim, criticTargetPath)
        self.actorTarget.requires_grad_(False)
        self.criticTarget.requires_grad_(False)
        self.actorTarget.softClone(self.actor, 1)
        self.criticTarget.softClone(self.critic, 1)
        
        self.actorOptimizer  = opt.Adam(params=self.actor.parameters(), lr=lrActor)
        self.criticOptimizer = opt.Adam(params=self.critic.parameters(), lr=lrCritic)        
        
    def remember(self, state, action, nextState, reward, isFinal):
        self.memory.store(state, action, nextState, reward, isFinal)
        
    def save(self, path):        
        self.actor.save(os.path.join(path, 'actorNet.pt'))
        self.actorTarget.save(os.path.join(path, 'criticNet.pt'))
        self.critic.save(os.path.join(path, 'actorTargetNet.pt'))
        self.criticTarget.save(os.path.join(path, 'criticTargetNet.pt'))
    
    def load(self, path):
        self.actor.load(os.path.join(path, 'actorNet.pt'))
        self.actorTarget.load(os.path.join(path, 'criticNet.pt'))
        self.critic.load(os.path.join(path, 'actorTargetNet.pt'))
        self.criticTarget.load(os.path.join(path, 'criticTargetNet.pt'))
        
    def saveCheckPoint(self):
        self.actor.saveCheckPoint()
        self.actorTarget.saveCheckPoint()
        self.critic.saveCheckPoint()
        self.criticTarget.saveCheckPoint()
    
    def loadCheckPoint(self):
        self.actor.loadCheckPoint()
        self.actorTarget.loadCheckPoint()
        self.critic.loadCheckPoint()
        self.criticTarget.loadCheckPoint()
        
    def act(self, state, withNoise=False):
        if not isinstance(state, tr.Tensor):
            state = tr.tensor(state)
        
        action = self.actor(state).detach()
        
        if withNoise:
            noise = self.noise()
            action += tr.tensor(noise, dtype=tr.float32)
            action = tr.clip(action, self.outBounds[0],self.outBounds[1])
        
        return action.numpy()
    
    def learn(self):
        if self.memory.count < self.batchSize:
            return float('nan'), float('nan')
        
        states, actions, nextStates, rewards, finals = self.memory.sample()
        
        
        states, actions, nextStates, rewards, finals = (
            tr.tensor(states, device=self.device),
            tr.tensor(actions, device=self.device),
            tr.tensor(nextStates, device=self.device),
            tr.tensor(rewards, device=self.device),
            tr.tensor(finals, device=self.device)
        )
        
        # Mise a jour des parametres du critique
        tragetActions = self.actorTarget.forward(nextStates)
        targetValues  = self.criticTarget.forward(nextStates, tragetActions).squeeze(1)
        values = self.critic(states, actions).squeeze(1)
        
        targets = rewards + self.discount * targetValues * (~finals)

        criticLoss = trf.mse_loss(values, targets)
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        # Mise a jour des parametres de l'acteur
        
        newActions = self.actor.forward(states)
        actorLosses  = self.critic.forward(states, newActions).squeeze(1)
        actorLoss  = -tr.mean(actorLosses)
        
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        self.actorOptimizer.step()
        
        self.actorTarget.softClone(self.actor, self.tau)
        self.criticTarget.softClone(self.critic, self.tau)
                
        return actorLoss.item(), criticLoss.item()