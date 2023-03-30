import os

import torch as tr
import torch.nn.functional as trf
import torch.optim as opt
import numpy as np

from Networks import CriticNetwork, ActorNetwork
from Memory import ReplayMemory


class Agent:
    def __init__(self, device, stateDim, actionDim, discount, tau, memoryCap, batchSize,
                    noise, lrActor, lrCritic, layersActor, layersCritic, outBounds,savePath):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.lrActor = lrActor
        self.lrCritic = lrCritic
        self.tau = tau
        self.discount = discount
        self.noise = noise
        self.savePath = savePath
        self.batchSize = batchSize
        self.outBounds = outBounds
        self.device = device
        
        actorPath  = os.path.join(savePath, 'actorNet.pt')
        criticPath = os.path.join(savePath, 'criticNet.pt')
        actorTargetPath  = os.path.join(savePath, 'actorTargetNet.pt')
        criticTargetPath = os.path.join(savePath, 'criticTargetNet.pt')
        
        self.memory = ReplayMemory (memoryCap, stateDim, actionDim, batchSize)
        self.actor  = ActorNetwork (device, stateDim, actionDim, *layersActor, outBounds, actorPath)
        self.critic = CriticNetwork(device, stateDim, actionDim, *layersCritic, criticPath) 
        
        self.actorTarget  = ActorNetwork (device, stateDim, actionDim, *layersActor, outBounds, actorTargetPath)
        self.criticTarget = CriticNetwork(device, stateDim, actionDim, *layersCritic, criticTargetPath)
        self.actorTarget.softClone(self.actor, 1)
        self.criticTarget.softClone(self.critic, 1)
        
        self.actorOptimizer  = opt.Adam(self.actor.parameters(), lrActor)
        self.criticOptimizer = opt.Adam(self.critic.parameters(), lrCritic)
        
        
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
        
        with tr.no_grad():
            action = self.actor(state)
            
            if withNoise:
                noise = tr.normal(mean = 0.0, std=self.noise, size=(self.actionDim,))
                action += noise
                tr.clamp_(action, min=self.outBounds[0],max=self.outBounds[1])
                    
        return action
    
    def learn(self):
        if self.memory.count < self.batchSize:
            return
        
        states, actions, nextStates, rewards, finals = self.memory.sample()
        states, actions, nextStates, rewards, finals = (
            tr.tensor(states, device=self.device),
            tr.tensor(actions, device=self.device),
            tr.tensor(nextStates, device=self.device),
            tr.tensor(rewards, device=self.device),
            tr.tensor(finals, device=self.device)

        )
        
        # Updating the critic network
        
        values = self.critic(states, actions).squeeze(1)
        tragetActions = self.actorTarget(states)
        targetValues  = self.criticTarget(states, tragetActions).squeeze(1)
        
        discountedValue = rewards + self.discount * targetValues * (~finals)
        
        self.criticOptimizer.zero_grad()
        criticLoss = trf.mse_loss(values, discountedValue)
        criticLoss.backward()
        self.criticOptimizer.step()

        
        # Updating the actor network
        #self.criticOptimizer.zero_grad()
        self.actorOptimizer.zero_grad()

        newActions = self.actor(states)
        actorLoss  = -self.critic(states, newActions).squeeze(1)
        actorLoss  = tr.mean(actorLoss)
        
        actorLoss.backward()
        self.actorOptimizer.step()
        
        self.actorTarget.softClone(self.actor, self.tau)
        self.criticTarget.softClone(self.critic, self.tau)
        
        
        
        