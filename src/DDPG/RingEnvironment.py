import os
import sys 
from dataclasses import dataclass


import gymnasium as gym
import numpy as np
import traci

import Vehicle as veh
from GenerateMaps import createRingFiles


tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

IDM_CONTROLLER_DEFAULT_PARAMS = {
    'a' :               5.0,
    'b' :               5.0,
    'T' :               1,
    'v' :               20.0,
    's' :               3.0,
    'imperfection' :    1/500
}

@dataclass
class _VehicleState:
    edgeID : str
    position : float
    speed : float
    leaderID : str
    leaderSpeed : float
    distanceToLeader : float
    followerID : str
    followerSpeed : float
    distanceToFollower : float
    acceleration  : float
    

class RingEnv(gym.Env):
    def __init__(self, 
                 radius,
                 numVehicles,
                 numVehiclesAuto,
                 maxSpeed = 25,
                 maxAccel = 5,
                 timeStep = 0.5,
                 maxIter = 1000,
                 IDMControllerParams = IDM_CONTROLLER_DEFAULT_PARAMS,
                 renderMode = 'none',
                 saveSpeeds = False,
                 generateFiles = False):        
        super().__init__()
        
        assert renderMode in ['human', 'none'] 
        veh.ResetIDCounter()
        self.radius = radius
        self.numVehicles = numVehicles
        self.numVehiclesAuto = numVehiclesAuto
        self.maxSpeed = maxSpeed
        self.maxAccel = maxAccel
        self.timeStep = timeStep
        self.render_mode = renderMode
        self.maxIter = maxIter
        self.IDMControllerParams = IDMControllerParams
        self.saveSpeeds = saveSpeeds
        self.edgeCount = 4
        self.currentIter = 0
        self.currentVehicleState = None
        
        self.trackLenght = radius*2*np.pi
        
        #self.autoSpeeds = np.empty((self.numVehiclesAuto, 100))
        
        outDir = os.path.join(os.getcwd(),'SumoFiles/')
        
        if generateFiles:
            createRingFiles(radius, self.edgeCount, 25, outDir)
        
        sumoBin = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
        if renderMode == 'human':
            sumoBin = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
        
        traci.start([sumoBin, '--step-length='+str(timeStep), '--no-warnings', '--no-step-log', '-c', os.path.join(outDir,'ring.sumocfg')])
        
        for i in range(self.edgeCount):
            traci.route.add('r'+str(i),['e'+str(i),'e'+str((i+1)%self.edgeCount)])
        
        self._PopulateRode()
        traci.simulationStep()
        
        self.metadata['render_modes'] = {'human','none'}
        self.observation_space = gym.spaces.Box(
            low =np.array([0.0]*(4*(numVehicles + numVehiclesAuto))),
            high=np.array([self.trackLenght, self.maxSpeed, self.maxAccel, np.inf]*(numVehicles+numVehiclesAuto))
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0]*numVehiclesAuto),
            high=np.array([self.maxSpeed]*numVehiclesAuto)
        )
    
    def step(self, action):
        assert len(action)==0 or (([0]*self.numVehiclesAuto <= action).any() and (action <= [self.maxSpeed]*self.numVehiclesAuto).any()), 'Inacceptable action: out bounds'
                
        for vhID,v in zip(self.autoVehiclesIndeces, action):
            self.vehicles[vhID].controller.setDesiredSpeed(v)
        
        accelerations = [v.controller.calcAcceleration(self.timeStep, s.speed, s.leaderSpeed, s.distanceToLeader)
                         for v,s in zip(self.vehicles, self.currentVehicleState) ]
        
        for vh,s,a in zip(self.vehicles, self.currentVehicleState, accelerations):
            v = s.speed + a*self.timeStep
            
            traci.vehicle.setSpeed(vh.id, max(v,0))
            s.acceleration = a
        
        traci.simulationStep()
        
        terminated = False
        truncated = False
        info = {}        
        
        collisions = traci.simulation.getCollisions()
        colliders = [c.collider for c in collisions]
        
        newVehicleState = [
            self._FetchState(self.vehicles[i]) if self.vehicles[i].id not in colliders 
            else self.currentVehicleState[i] 
            for i in range(self.numVehicles)
        ]
        positions = np.array([s.speed for s in newVehicleState ], dtype=np.float32)
        speeds = np.array([s.speed for s in newVehicleState ], dtype=np.float32)
        accelerations = np.array([s.speed for s in newVehicleState ], dtype=np.float32)
        distances = np.array([s.distanceToLeader for s in newVehicleState ], dtype=np.float32)
        
        #self.autoSpeeds[:,self.currentIter % 100] = speeds[:self.numVehiclesAuto]
        
        reward = 0
        desiredSpeed = 20
        
        if collisions:
            reward -= 1000
            terminated = True
       
        reward -= 0.03 * np.mean(np.abs(desiredSpeed-speeds[self.autoVehiclesIndeces])) * 1000 / self.maxIter
        reward -= 0.01 * np.mean(np.abs(desiredSpeed-speeds)) * 1000 / self.maxIter
        reward -= 0.0001 * np.mean(np.abs(accelerations)) * 1000 / self.maxIter
        
        
        self.currentVehicleState = newVehicleState
        self.currentIter += 1
        
        state = np.empty(((self.numVehicles + self.numVehiclesAuto)* 4,),dtype=np.float32)
        state[0:self.numVehiclesAuto*4:4] = positions[self.autoVehiclesIndeces]
        state[1:self.numVehiclesAuto*4:4] = speeds[self.autoVehiclesIndeces]
        state[2:self.numVehiclesAuto*4:4] = accelerations[self.autoVehiclesIndeces]
        state[3:self.numVehiclesAuto*4:4] = distances[self.autoVehiclesIndeces]
        state[self.numVehiclesAuto*4 + 0::4] = positions
        state[self.numVehiclesAuto*4 + 1::4] = speeds
        state[self.numVehiclesAuto*4 + 2::4] = accelerations
        state[self.numVehiclesAuto*4 + 3::4] = distances
              
        
        if self.currentIter >= self.maxIter:
            truncated = True
        
        return state, reward, terminated, truncated, info
    
    def reset(self):
        for v in self.vehicles:
            traci.vehicle.remove(v.id)
        self.autoSpeeds = np.empty((self.numVehiclesAuto, 100))

        traci.simulationStep()
        veh.ResetIDCounter()
        self._PopulateRode()
        traci.simulationStep()

        self.currentIter = 0
        self.currentVehicleState = [self._FetchState(v) for v in self.vehicles]
        
        state = np.array([ [s.position,s.speed,s.acceleration,s.distanceToLeader] for s in self.currentVehicleState ], dtype=np.float32).reshape(-1)
        
        autoState = np.empty((4*self.numVehiclesAuto,), dtype=np.float32)
        for i,j in enumerate(self.autoVehiclesIndeces):
            autoState[4*i+0] = state[4*j+0] 
            autoState[4*i+1] = state[4*j+1] 
            autoState[4*i+2] = state[4*j+2] 
            autoState[4*i+3] = state[4*j+3] 
            
        return np.concatenate((autoState,state)), {}
    
    def close(self):
        traci.close()
        
    def _getStartPos(self,edge):
        edgeLenght = self.trackLenght / self.edgeCount
        orderedEdges = ['e0',':n0_0','e1',':n1_0','e2',':n2_0','e3',':n3_0']
        orderedLeghts = [edgeLenght,0.1,edgeLenght,0.1,edgeLenght,0.1,edgeLenght,0.1]
        pos = 0
        i = 0
        while edge != orderedEdges[i]:
            pos += orderedLeghts[i]
            i+=1
        return pos
    
    def _FetchState(self, vehicle):
        edgeID = traci.vehicle.getRoadID(vehicle.id)
        speed = max(0,traci.vehicle.getSpeed(vehicle.id))
        position = traci.vehicle.getLanePosition(vehicle.id)
        position += self._getStartPos(edgeID)
        
        leader = traci.vehicle.getLeader(vehicle.id, dist= 1000)
        leaderID, distanceLeader,leaderSpeed = '',  float('inf'), 0
        if leader:
            leaderID, distanceLeader = leader[0] , max(0,leader[1])
            leaderSpeed = traci.vehicle.getSpeed(leaderID)
            
        follower = traci.vehicle.getFollower(vehicle.id, dist= 1000)
        followerID, distanceFollower,followerSpeed = '', float('inf'), 0
        if follower[0] != '':
            followerID, distanceFollower = follower[0] , max(0,follower[1])
            followerSpeed = traci.vehicle.getSpeed(followerID)
            
        return _VehicleState( edgeID=edgeID,
                              position=position,
                              speed=speed,
                              leaderID=leaderID,
                              leaderSpeed=max(leaderSpeed,0),
                              distanceToLeader=np.clip(distanceLeader,0,1000),
                              followerID=followerID,
                              followerSpeed=max(followerSpeed,0),
                              distanceToFollower=np.clip(distanceFollower,0,1000),
                              acceleration=0)
        
    def _PopulateRode(self):
        
        edgeLenght = self.trackLenght / self.edgeCount
        separation = self.trackLenght / self.numVehicles
        self.vehicles = []
        self.autoVehiclesIndeces = []
                
        for i in range(self.numVehicles):
            pos = i*separation
            edge = int(pos / edgeLenght)
            pos -= edge*edgeLenght
            
            v = veh.Vehicle(veh.IDMController(**self.IDMControllerParams), None, self.saveSpeeds)
            self.vehicles.append(v)
            traci.vehicle.add(v.id, 'r'+str(edge), departPos=pos)
            traci.vehicle.setSpeedMode(v.id, 0)
        
        firstAuto = np.random.randint(0, self.numVehicles)
        
        for i in range(self.numVehiclesAuto):
            j = int(i*(self.numVehicles/self.numVehiclesAuto) + firstAuto) % self.numVehicles
            v = self.vehicles[j]
            v.controller = veh.ObedientController(self.maxAccel)
            self.autoVehiclesIndeces.append(j)
            traci.vehicle.setColor(v.id, (20,40,230))
        self.autoVehiclesIndeces.sort()