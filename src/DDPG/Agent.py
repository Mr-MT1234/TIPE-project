import os

import torch as tr
import torch.nn as trn
import torch.nn.functional as trf
import torch.optim as opt
import numpy as np

from Networks import CriticNetwork, ActorNetwork
from Memory import ReplayMemory


class Agent:
    def __init__(self, device, stateDim, actionDim, discount, tau, memoryCap, batchSize,
                    noise, lrActor, lrCritic, layersActor, layersCritic, outBounds, learningDecay, noiseDecay,savePath):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.lrActor = lrActor
        self.lrCritic = lrCritic
        self.tau = tau
        self.discount = discount
        self.noise = noise
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
        
        self.memory = ReplayMemory (memoryCap, stateDim, actionDim, batchSize)
        self.actor  = ActorNetwork (device, stateDim, actionDim, *layersActor, outBounds, actorPath)
        self.critic = CriticNetwork(device, stateDim, actionDim, *layersCritic, criticPath) 
        
        self.actorTarget  = ActorNetwork (device, stateDim, actionDim, *layersActor, outBounds, actorTargetPath)
        self.criticTarget = CriticNetwork(device, stateDim, actionDim, *layersCritic, criticTargetPath)
        self.actorTarget.requires_grad_(False)
        self.criticTarget.requires_grad_(False)
        self.actorTarget.softClone(self.actor, 1)
        self.criticTarget.softClone(self.critic, 1)
        
        self.actorOptimizer  = opt.Adam(params=self.actor.parameters(), lr=lrActor,  eps=1e-7)
        self.criticOptimizer = opt.Adam(params=self.critic.parameters(), lr=lrCritic, eps=1e-7)
        
        self.mse = trn.MSELoss()
        
        
    def remember(self, state, action, nextState, reward, isFinal):
        self.memory.store(state, action, nextState, reward, isFinal)
        
    def save(self):
        self.actor.save()
        self.actorTarget.save()
        self.critic.save()
        self.criticTarget.save()
    
    def load(self):
        self.actor.load()
        self.actorTarget.load()
        self.critic.load()
        self.criticTarget.load()
        
    def act(self, state, withNoise=False):
        if not isinstance(state, tr.Tensor):
            state = tr.tensor(state)
        
        action = self.actor(state).detach()
        
        if withNoise:
            noise = np.random.normal(0, self.noise, self.actionDim)
            action += tr.tensor(noise, device=self.device)
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
        
        # Updating the critic network
        tragetActions = self.actorTarget.forward(nextStates)
        targetValues  = self.criticTarget.forward(nextStates, tragetActions).squeeze(1)
        values = self.critic(states, actions).squeeze(1)
        
        targets = rewards + self.discount * targetValues * (~finals)

        criticLoss = self.mse(values, targets)
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        # Updating the actor network
        #self.criticOptimizer.zero_grad()
        
        newActions = self.actor.forward(states)
        actorLosses  = self.critic.forward(states, newActions).squeeze(1)
        actorLoss  = -tr.mean(actorLosses)
        
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        self.actorOptimizer.step()
        
        self.actorTarget.softClone(self.actor, self.tau)
        self.criticTarget.softClone(self.critic, self.tau)
                
        return actorLoss.detach().numpy(), criticLoss.detach().numpy()
    
    def decay(self):
        self.actorOptimizer.param_groups[0]['lr'] *= self.learningDecay
        self.criticOptimizer.param_groups[0]['lr'] *= self.learningDecay
        self.noise *= self.noiseDecay