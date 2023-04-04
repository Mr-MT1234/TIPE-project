import torch as tr
import torch.nn as trn
import torch.nn.functional as trf
import numpy as np

class CriticNetwork(trn.Module):
    def __init__(self, device, stateDim, actionDim, l1, l2,savePath):
        super(CriticNetwork, self).__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.savePath = savePath
        self.layer1 = trn.Linear(stateDim, l1, device=device)
        self.layer21 = trn.Linear(l1, l2, device=device)
        self.layer22 = trn.Linear(actionDim, l2, device=device)
        self.output = trn.Linear(l2,1, device=device)
        
        """
        self.seq = trn.Sequential(
            trn.Linear(stateDim + actionDim, l1, device=device),
            trn.ReLU(),
            trn.Linear(l1, l2, device=device),
            trn.ReLU(),
            trn.Linear(l2,1, device=device)
        )
        """
        
    def forward(self, state, action):
        x = self.layer1(state)
        x = trf.relu(x)
        x = self.layer21(x)
        y = self.layer22(action)
        
        out = trf.relu(x+y)
        out = self.output(out)
        return out
        """
        input = tr.cat((state, action), 1)
        return self.seq(input)
        """
        
            
    def softClone(self, other, tau):
        with tr.no_grad():
            for x, y in zip(self.parameters(), other.parameters()):
                x.data.mul_(1-tau)
                x.data.add_(tau * y.data)
    
    def save(self):
        tr.save(self.state_dict(), self.savePath)
        
    def load(self):
        stateDict = tr.load(self.savePath)
        self.load_state_dict(stateDict)

        
class ActorNetwork(trn.Module):
    def __init__(self, device, stateDim, actionDim, l1, l2, bounds, savePath):
        super(ActorNetwork, self).__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.bounds = tr.tensor(bounds, device=device)
        self.savePath = savePath

        self.layer1 = trn.Linear(stateDim, l1, device=device)
        self.layer2 = trn.Linear(l1, l2, device=device)
        self.output = trn.Linear(l2, actionDim, device=device)
        """
        self.seq = trn.Sequential(
            trn.Linear(stateDim, l1, device=device),
            trn.ReLU(),
            trn.Linear(l1, l2, device=device),
            trn.ReLU(),
            trn.Linear(l2,actionDim, device=device),
            trn.Tanh()
        )
        
        """
        
    def forward(self, state):
        x = self.layer1(state)
        x = trf.relu(x)
        x = self.layer2(x)
        x = trf.relu(x)
        x = self.output(x)
        x = trf.tanh(x)
        """
        x = self.seq(state)
        
        """
        x = ((x + 1) / 2) * (self.bounds[1] -  self.bounds[0]) + self.bounds[0]
        return x
    
        
    def softClone(self, other, tau):
        with tr.no_grad():
            for x, y in zip(self.parameters(), other.parameters()):
                x.data.mul_(1-tau)
                x.data.add_(tau * y.data)
    
    def save(self):
        tr.save(self.state_dict(), self.savePath)
        
    def load(self):
        stateDict = tr.load(self.savePath)
        self.load_state_dict(stateDict)
