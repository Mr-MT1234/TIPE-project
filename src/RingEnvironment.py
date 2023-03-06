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

__MaxSpeed = 34
__AccidentDistance = 0.3
__IDMControllerParams = {
    'a' :               5.0,
    'b' :               5.0,
    'T' :               2.0,
    'v' :               20.0,
    's' :               40.0,
    'imperfection' :    1/500
}

ACTIONS = ['Decelerate', 'NoOp', 'Accelerate']

@dataclass
class __VehicleState:
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
        super().__init__()
        
        assert renderMode in ['human', 'none'] 
        
        self.numVehicles = numVehicles
        self.radius = radius
        self.selfDrivingNum = int(selfDrivingPercentage * numVehicles)
        self.selfDrivingAccel = selfDrivingAccel
        self.accidentPunishment = accidentPunishmeent
        self.metadata['render_modes'] = {'human','none'}
        self.action_space = gym.spaces.MultiDiscrete([3]*self.selfDrivingNum)
        self.observation_space = gym.spaces.Box(
            low= np.array([0.0]*(5*numVehicles)),
            high=np.array([__MaxSpeed,__MaxSpeed,__MaxSpeed,float('inf'),float('inf')]*numVehicles)
        )
        
        self.render_mode = renderMode
        
        edgeCount = 4
        res = int(radius * 2)
        outputDir = os.pathjoin(os.getcwd(), '/sumoFiles/')
        createRingFiles(radius, edgeCount, res, outputDir)
        
        sumoBin = 'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui' # May change
        if renderMode == 'none': 
            sumoBin = 'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo' # May change
            
        traci.start([sumoBin, '--step-length='+str(timeStep),"-c",os.path.join(outputDir('/Ring.sumocfg'))])
        
        edgeLength = radius * 2 * np.pi / edgeCount
        carsPerEdge = numVehicles / edgeCount
        separation = edgeLength / carsPerEdge
        self.vehicles = []
        self.autoVehicles = []
        
        assert separation > 10
        
        for i in range(edgeCount):
            traci.route.add('r'+str(i),['e'+str(i),'e'+str((i+1)%edgeCount)])
        
        for j in range(numVehicles):
            for i in range(np.ceil(carsPerEdge)):
                    v = veh.Vehicle(veh.IDMController(**__IDMControllerParams), None, saveSpeeds)
                    self.vehicles.append(v)
                    traci.vehicle.add(v.id, 'r'+str(j), departPos=i*separation)
                    traci.vehicle.setSpeedMode(v.id, 0)
                    
        for v in self.vehicles[::int(1/selfDrivingPercentage)]:
            v.controller = veh.ObedientController()
            self.autoVehicles.append(v)
                    
        
    def render(self):
        pass
    
    def step(self, action):
        accelerations = [(a - 1)*self.selfDrivingAccel for a in action]
        states = [self.__FetchState(v.id) for v in self.vehicles] 
        
        for v,a in zip(self.autoVehicles, accelerations):
            v.controller.setNextAcceleration(a)
        
        for v,s in zip(self.vehicles, states):
            a = v.controller.calcAcceleration(self.timeStep, s.speed, s.leaderSpeed, s.distanceToLeader)
            v = s.speed + a*self.timeStep
            traci.vehicle.setSpeed(v.id, v)
        
        traci.simulationStep()
        
        terminated = False
        truncated = False
        info = {}
        done = False
        
        
        newStates = [self.__FetchState(v) for v in self.vehicles]
        newStatesAuto = [self.__FetchState(v) for v in self.autoVehicles]
        
        observation = np.array([ [s.speed, s.leaderSpeed, s.FollowerSpeed, s.distanceToLeader, s.distanceToFollower] for s in newStatesAuto ])
        np.reshape(observation,-1)
        
        reward = 0
        for s in newStatesAuto:
            if  s.distanceToLeader   < __AccidentDistance \
             or s.distanceToFollower < __AccidentDistance:
                 reward -= self.accidentPunishment
                 terminated = done = True
        
        reward += sum(newStates, key=lambda x: x.speed) / len(newStates)
                
        return observation, reward, terminated, truncated, info, done
    
    def reset(self):
        traci.close()
        
        sumoBin = 'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui' # May change
        if self.renderMode == 'none': 
            sumoBin = 'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo' # May change
        
        traci.start([sumoBin, '--step-length='+str(self.timeStep),"-c",os.path.join(self.outputDir('/Ring.sumocfg'))])
        
        edgeCount = 4
        edgeLength = self.radius * 2 * np.pi / edgeCount
        carsPerEdge = self.numVehicles / edgeCount
        separation = edgeLength / carsPerEdge
        self.vehicles = []
        self.autoVehicles = []
        
        assert separation > 10
        
        for i in range(edgeCount):
            traci.route.add('r'+str(i),['e'+str(i),'e'+str((i+1)%edgeCount)])
        
        for j in range(self.numVehicles):
            for i in range(np.ceil(carsPerEdge)):
                    v = veh.Vehicle(veh.IDMController(**__IDMControllerParams), None, self.saveSpeeds)
                    self.vehicles.append(v)
                    traci.vehicle.add(v.id, 'r'+str(j), departPos=i*separation)
                    traci.vehicle.setSpeedMode(v.id, 0)
                    
        for v in self.vehicles[::int(1/self.selfDrivingPercentage)]:
            v.controller = veh.ObedientController()
            self.autoVehicles.append(v)
        
    
    def close(self):
        traci.close()
    
    def __FetchState(vehicle):
        edgeID = traci.vehicle.getRoadID(vehicle.id)
        speed = traci.vehicle.getSpeed(vehicle.id)
        position = traci.vehicle.getLanePosition(vehicle.id)
        
        leader = traci.vehicle.getLeader(vehicle.id)
        leaderID, distanceLeader,leaderSpeed = '', float('-inf'), float('-inf')
        if leader:
            leaderID, distanceLeader = leader[0] , max(0,leader[1])
            leaderSpeed = traci.vehicle.getSpeed(leaderID)
            
        follower = traci.vehicle.getFollower(vehicle.id)
        followerID, distanceFollower,followerSpeed = '', float('-inf'), float('-inf')
        if follower:
            followerID, distanceFollower = leader[0] , max(0,leader[1])
            followerSpeed = traci.vehicle.getSpeed(leaderID)
            
        return __VehicleState(edgeID=edgeID,
                              position=position,
                              speed=speed,
                              leaderID=leaderID,
                              leaderSpeed=leaderSpeed,
                              distanceToLeader=distanceLeader,
                              followerID=followerID,
                              followerSpeed=followerSpeed,
                              distanceToFollower=distanceFollower)
