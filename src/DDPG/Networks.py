import torch as tr
import torch.nn as trn
import torch.nn.functional as trf
import numpy as np

class CriticNetwork(trn.Module):
    def __init__(self, device, stateDim, actionDim, l1, l2,savePath):
        super(CriticNetwork, self).__init__()
        self.stateDim = stateDim[0]
        self.actionDim = actionDim
        self.savePath = savePath
        self.layer1 = trn.Linear(self.stateDim, l1, device=device)
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
    
    def saveCheckPoint(self):
        tr.save(self.state_dict(), self.savePath)
        
    def loadCheckPoint(self):
        stateDict = tr.load(self.savePath)
        self.load_state_dict(stateDict)
        
    def save(self, path):
        tr.save(self.state_dict(), path)
    
    def load(self,path):
        stateDict = tr.load(path)
        self.load_state_dict(stateDict)
    

class ActorNetwork(trn.Module):
    def __init__(self, device, stateDim, actionDim, l1, l2, bounds, savePath):
        super(ActorNetwork, self).__init__()
        self.stateDim = stateDim[0]
        self.actionDim = actionDim
        self.bounds = tr.tensor(bounds, device=device)
        self.savePath = savePath

        self.layer1 = trn.Linear(self.stateDim, l1, device=device)
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
    
    def saveCheckPoint(self):
        tr.save(self.state_dict(), self.savePath)
        
    def loadCheckPoint(self):
        stateDict = tr.load(self.savePath)
        self.load_state_dict(stateDict)
        
    def save(self, path):
        tr.save(self.state_dict(), path)
    
    def load(self,path):
        stateDict = tr.load(path)
        self.load_state_dict(stateDict)
    
        
class CriticConvNetwork(trn.Module):
    def __init__(self, device, stateShape, actionDim,savePath):
        super(CriticConvNetwork, self).__init__()
        self.stateShape = stateShape
        self.actionDim = actionDim
        self.savePath = savePath

        self.seqState = trn.Sequential(
            trn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3,5,1)),
            trn.MaxPool3d(kernel_size=(1,2,1), stride=(1,2,1)),
            trn.ReLU(),
            trn.Conv3d(in_channels=4, out_channels=16, kernel_size=(3,3,2), groups=4),
            trn.MaxPool3d(kernel_size=(1,2,1), stride=(1,2,1)),
            trn.ReLU(),
            
            trn.Flatten(-4,-1),
            
            trn.LazyLinear(out_features=256),
            trn.ReLU(),
        ).to(device)
        
        dummy = tr.tensor(np.empty((1,*stateShape),dtype=np.float32))
        self.seqState(dummy)
        
        self.linearAction = trn.Linear(in_features=actionDim,out_features=256, device=device)
        self.output = trn.Sequential(
            trn.Linear(in_features=256,out_features=256, device=device),
            trn.Linear(in_features=256,out_features=1, device=device)
        )
        
    def forward(self, state, action):
        x = self.seqState(state)
        y = self.linearAction(action)
        out = self.output(trf.relu(x+y))
        return out
            
    def softClone(self, other, tau):
        with tr.no_grad():
            for x, y in zip(self.parameters(), other.parameters()):
                x.data.mul_(1-tau)
                x.data.add_(tau * y.data)
    
    def saveCheckPoint(self):
        tr.save(self.state_dict(), self.savePath)
        
    def loadCheckPoint(self):
        stateDict = tr.load(self.savePath)
        self.load_state_dict(stateDict)
        
    def save(self, path):
        tr.save(self.state_dict(), path)
    
    def load(self,path):
        stateDict = tr.load(path)
        self.load_state_dict(stateDict)
    
    
class ActorConvNetwork(trn.Module):
    def __init__(self, device, stateShape, actionDim, outBounds, savePath):
        super(ActorConvNetwork, self).__init__()
        self.stateShape = stateShape
        self.actionDim = actionDim
        self.outBounds = outBounds
        self.savePath = savePath
        
        self.seq = trn.Sequential(
            trn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3,5,1)),
            trn.MaxPool3d(kernel_size=(1,2,1), stride=(1,2,1)),
            trn.ReLU(),
            trn.Conv3d(in_channels=4, out_channels=16, kernel_size=(3,3,2), groups=4),
            trn.MaxPool3d(kernel_size=(1,2,1), stride=(1,2,1)),
            trn.ReLU(),
            
            trn.Flatten(-4,-1),
            
            trn.LazyLinear(out_features=256),
            trn.ReLU(),
            trn.Linear(in_features=256, out_features=256),
            trn.ReLU(),
            trn.Linear(in_features=256,out_features=actionDim),
            trn.Tanh(),
        ).to(device)
        
        dummy = tr.tensor(np.empty((1,*stateShape),dtype=np.float32))
        self.seq(dummy)
        
    def forward(self, state):    
        out = self.seq(state)
        return ((out + 1) / 2) * tr.tensor((self.outBounds[1] - self.outBounds[0]) + self.outBounds[0],dtype=tr.float32)
            
    def softClone(self, other, tau):
        with tr.no_grad():
            for x, y in zip(self.parameters(), other.parameters()):
                x.data.mul_(1-tau)
                x.data.add_(tau * y.data)
    
    def saveCheckPoint(self):
        tr.save(self.state_dict(), self.savePath)
        
    def loadCheckPoint(self):
        stateDict = tr.load(self.savePath)
        self.load_state_dict(stateDict)

    def save(self, path):
        tr.save(self.state_dict(), path)
    
    def load(self,path):
        stateDict = tr.load(path)
        self.load_state_dict(stateDict)
    