import math
import os
import sys 
import threading
import time
from dataclasses import dataclass


import gymnasium as gym
import gym.spaces
import numpy as np
import traci

import Vehicle as veh
from GenerateTrafficRing import createRingFiles


tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

MaxSpeed = 34
AccidentDistance = 1
IDMControllerParams = {
    'a' :               5.0,
    'b' :               5.0,
    'T' :               2.0,
    'v' :               20.0,
    's' :               40.0,
    'imperfection' :    1/500
}

ACTIONS = ['Decelerate', 'NoOp', 'Accelerate']

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
    

class RingEnv(gym.Env):
    
    def __init__(self, *, 
                 radius,
                 numVehicles,
                 selfDrivingPercentage = 0.05,
                 timeStep = 0.5,
                 saveSpeeds = False,
                 selfDrivingAccel = 2,
                 accidentPunishmeent = -1000,
                 renderMode = 'human'):
        global MaxSpeed
        
        super().__init__()
        
        assert renderMode in ['human', 'none'] 
        
        self.numVehicles = numVehicles
        self.radius = radius
        self.selfDrivingNum = int(selfDrivingPercentage * numVehicles)
        self.selfDrivingAccel = selfDrivingAccel
        self.accidentPunishment = accidentPunishmeent
        self.saveSpeeds = saveSpeeds
        self.selfDrivingPercentage = selfDrivingPercentage
        self.timeStep = timeStep
        self.metadata['render_modes'] = {'human','none'}
        self.action_space = gym.spaces.MultiDiscrete([3]*self.selfDrivingNum)
        self.observation_space = gym.spaces.Box(
            low= np.array([0.0]*(5*numVehicles)),
            high=np.array([MaxSpeed,MaxSpeed,MaxSpeed,float('inf'),float('inf')]*numVehicles)
        )
        
        self.render_mode = renderMode
        
        sumoBin = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo"
        if renderMode == 'human':
            sumoBin = 'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui'
        
        traci.start([sumoBin, '--step-length='+str(timeStep),"-c", 'C:/Users/Tribik/Documents/Mohamed/Études/MP/TIPE/Code/Ring/SumoFiles/ring.sumocfg'])
        
        self.__PopulateRode()
        traci.simulationStep()
        
    def render(self):
        pass
    
    def step(self, action):
        global AccidentDistance
        
        accelerationsAuto = [(a - 1)*self.selfDrivingAccel for a in action]
        states = [self.__FetchState(v) for v in self.vehicles] 
        
        for v,a in zip(self.autoVehicles, accelerationsAuto):
            v.controller.setNextAcceleration(a)
        
        accelerations = [v.controller.calcAcceleration(self.timeStep, s.speed, s.leaderSpeed, s.distanceToLeader)
                         for v,s in zip(self.vehicles, states) ]
        
        for vh,s,a in zip(self.vehicles, states,accelerations):
            v = s.speed + a*self.timeStep
            traci.vehicle.setSpeed(vh.id, v)
        
        traci.simulationStep()
        
        terminated = False
        truncated = False
        info = {}
        done = False
        
        newStates = [self.__FetchState(v) for v in self.vehicles]
        newStatesAuto = [self.__FetchState(v) for v in self.autoVehicles]
        
        observation = np.array([ [s.speed, s.leaderSpeed, s.followerSpeed, s.distanceToLeader, s.distanceToFollower] for s in newStatesAuto ])
        np.reshape(observation,-1)
        
        reward = 0
        for s in newStatesAuto:
            if  s.distanceToLeader   < AccidentDistance \
             or s.distanceToFollower < AccidentDistance:
                 reward -= self.accidentPunishment
                 terminated = done = True
        
        reward += np.mean(np.abs(accelerations))
                
        return observation, reward, terminated, truncated, info, done
    
    def reset(self):
        for v in self.vehicles:
            traci.vehicle.remove(v.id)

        self.__PopulateRode()
        traci.simulationStep()
    
    def close(self):
        traci.close()
    
    def __FetchState(self, vehicle):
        edgeID = traci.vehicle.getRoadID(vehicle.id)
        speed = traci.vehicle.getSpeed(vehicle.id)
        position = traci.vehicle.getLanePosition(vehicle.id)
        
        leader = traci.vehicle.getLeader(vehicle.id, dist= 1000)
        leaderID, distanceLeader,leaderSpeed = '', float('-inf'), float('-inf')
        if leader:
            leaderID, distanceLeader = leader[0] , max(0,leader[1])
            leaderSpeed = traci.vehicle.getSpeed(leaderID)
            
        follower = traci.vehicle.getFollower(vehicle.id, dist= 1000)
        followerID, distanceFollower,followerSpeed = '', float('-inf'), float('-inf')
        if follower:
            followerID, distanceFollower = follower[0] , max(0,follower[1])
            followerSpeed = traci.vehicle.getSpeed(followerID)
            
        return _VehicleState( edgeID=edgeID,
                              position=position,
                              speed=speed,
                              leaderID=leaderID,
                              leaderSpeed=leaderSpeed,
                              distanceToLeader=distanceLeader,
                              followerID=followerID,
                              followerSpeed=followerSpeed,
                              distanceToFollower=distanceFollower)
        
    def __PopulateRode(self):
        global IDMControllerParams
        
        edgeCount = 4
        edgeLength = self.radius * 2 * np.pi / edgeCount
        carsPerEdge = self.numVehicles / edgeCount
        separation = edgeLength / carsPerEdge
        self.vehicles = []
        self.autoVehicles = []
        
        assert separation > 5
        
        for i in range(edgeCount):
            traci.route.add('r'+str(i),['e'+str(i),'e'+str((i+1)%edgeCount)])
        
        for j in range(edgeCount):
            for i in range(round(carsPerEdge)):
                    v = veh.Vehicle(veh.IDMController(**IDMControllerParams), None, self.saveSpeeds)
                    self.vehicles.append(v)
                    traci.vehicle.add(v.id, 'r'+str(j), departPos=i*separation)
                    traci.vehicle.setSpeedMode(v.id, 0)
                    
        for v in self.vehicles[::int(1/self.selfDrivingPercentage)]:
            v.controller = veh.ObedientController()
            self.autoVehicles.append(v)
