import numpy as np

class ReplayMemory:
    def __init__(self, capacity, stateShape, actionDim, batchSize):
        self.capacity  = capacity
        self.stateShape  = stateShape
        self.actionDim = actionDim
        self.batchSize = batchSize
        
        self.states     = np.empty((capacity, *stateShape) , dtype=np.float32)
        self.actions    = np.empty((capacity, actionDim), dtype=np.float32)
        self.nextStates = np.empty((capacity, *stateShape) , dtype=np.float32)
        self.rewards    = np.empty((capacity, ), dtype=np.float32)
        self.isFinals   = np.empty((capacity, ), dtype=bool)
        self.count = 0
    
    def store(self, state, action, nextState, reward, isFinal):
        i = self.count % self.capacity
        
        self.states[i] = state     
        self.actions[i] = action    
        self.nextStates[i] = nextState 
        self.rewards[i] = reward    
        self.isFinals[i] = isFinal
        
        self.count += 1
        
    def sample(self):
        assert self.batchSize <= self.count , 'Not enough data'
        
        batchIndeces = np.random.choice(self.count, size=self.batchSize, replace=False)
        
        return (
                self.states[batchIndeces],
                self.actions[batchIndeces],    
                self.nextStates[batchIndeces], 
                self.rewards[batchIndeces],    
                self.isFinals[batchIndeces]
        )
        
    
        
        