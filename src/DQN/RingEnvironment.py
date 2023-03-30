import math
import os
import sys 
import threading
import time
import multiprocessing
from dataclasses import dataclass


import gymnasium as gym
import gymnasium.spaces
import numpy as np
import traci

import Vehicle as veh
from GenerateTrafficRing import createRingFiles


tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

MaxSpeed = 34
IDMControllerParams = {
    'a' :               2.0,
    'b' :               2.0,
    'T' :               1,
    'v' :               20.0,
    's' :               3.0,
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
    acceleration : float
    

class RingEnv(gym.Env):
    
    def __init__(self, *, 
                 radius,
                 numVehicles,
                 selfDrivingPercentage = 0.05,
                 timeStep = 0.5,
                 saveSpeeds = False,
                 selfDrivingAccel = 2,
                 accidentPunishmeent = -1000,
                 maxIter = 1000,
                 renderMode = 'human',
                 generateFiles = False):
        global MaxSpeed
        
        super().__init__()
        
        assert renderMode in ['human', 'none'] 
        veh.ResetIDCounter()
        self.numVehicles = numVehicles
        self.radius = radius
        self.selfDrivingAccel = selfDrivingAccel
        self.accidentPunishment = accidentPunishmeent
        self.saveSpeeds = saveSpeeds
        self.selfDrivingPercentage = selfDrivingPercentage
        self.timeStep = timeStep
        self.edgeCount = 4
        self.render_mode = renderMode
        self.maxIter = maxIter
        self.currentIter = 0
        self.collision = False
        
        outDir = os.path.join(os.getcwd(),'SumoFiles/')
        
        if generateFiles:
            createRingFiles(radius, 4, 25, outDir)
        
        sumoBin = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
        if renderMode == 'human':
            sumoBin = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
        
        traci.start([sumoBin, '--step-length='+str(timeStep), '--no-warnings', '--no-step-log', '-c', os.path.join(outDir,'ring.sumocfg')])
        
        for i in range(self.edgeCount):
            traci.route.add('r'+str(i),['e'+str(i),'e'+str((i+1)%self.edgeCount)])
        
        self.__PopulateRode()
        traci.simulationStep()
        
        self.selfDrivingNum = len(self.autoVehicles)
        self.metadata['render_modes'] = {'human','none'}
        self.action_space = gym.spaces.Discrete(3**self.selfDrivingNum)
        self.observation_space = gym.spaces.Box(
            low= np.array([0.0]*(5*numVehicles)),
            high=np.array([MaxSpeed,MaxSpeed,MaxSpeed,float('inf'),float('inf')]*numVehicles)
        )
        
        
    def render(self):
        pass
    
    def step(self, action):
        if self.collision:
            return [0]*(5*self.selfDrivingNum), self.accidentPunishment, False, True, {}, False
        action = action.item()
        actions = []
        
        while action > 0:
            a = action % 3
            action //= 3
            actions.append(a)
        
        actions += [0]*(self.selfDrivingNum - len(actions))
                
        accelerationsAuto = [(a - 1)*self.selfDrivingAccel for a in actions]
        states = [self.__FetchState(v) for v in self.vehicles] 
        statesAuto = [self.__FetchState(v) for v in self.autoVehicles] 
        
        for v,a in zip(self.autoVehicles, accelerationsAuto):
            v.controller.setNextAcceleration(a)
        
        accelerations = [v.controller.calcAcceleration(self.timeStep, s.speed, s.leaderSpeed, s.distanceToLeader)
                         for v,s in zip(self.vehicles, states) ]
        
        for vh,s,a in zip(self.vehicles, states, accelerations):
            v = s.speed + a*self.timeStep
            
            traci.vehicle.setSpeed(vh.id, max(v,0))
            s.acceleration = a
        
        traci.simulationStep()
        
        terminated = False
        truncated = False
        info = {}
        done = False
        
        
        collisions = traci.simulation.getCollisions()
        
        newStates = [self.__FetchState(v) for v in self.vehicles]
        newStatesAuto = [self.__FetchState(v) for v in self.autoVehicles]
        
        reward = self.__CalculateReward3(states, statesAuto,newStates,newStatesAuto, collisions)
        
        observation = np.array([ [s.speed, s.leaderSpeed, s.followerSpeed, s.distanceToLeader, s.distanceToFollower] for s in newStatesAuto ]).reshape(-1)
        np.reshape(observation,-1)
        
        if self.currentIter >= self.maxIter:
            terminated = True
        self.currentIter += 1
        
        if  collisions:
            terminated = done = False
            self.collision = True
            return [0]*(5*self.selfDrivingNum), reward, terminated, truncated, info, done
                
        return observation, reward, terminated, truncated, info, done
    
    def reset(self):
        for v in self.vehicles:
            traci.vehicle.remove(v.id)
        traci.simulationStep()
        veh.ResetIDCounter()
        self.__PopulateRode()
        traci.simulationStep()
        self.currentIter = 0
        self.collision = False
        newStatesAuto = [self.__FetchState(v) for v in self.autoVehicles]
        observation = np.array([ [s.speed, s.leaderSpeed, s.followerSpeed, s.distanceToLeader, s.distanceToFollower] for s in newStatesAuto ]).reshape(-1)
        return observation, {}
    
    def close(self):
        traci.close()
    
    def __FetchState(self, vehicle):
        edgeID = traci.vehicle.getRoadID(vehicle.id)
        speed = max(0,traci.vehicle.getSpeed(vehicle.id))
        position = traci.vehicle.getLanePosition(vehicle.id)
        
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
        
    def __PopulateRode(self):
        global IDMControllerParams
        
        self.edgeCount = 4
        edgeLength = self.radius * 2 * np.pi / self.edgeCount
        carsPerEdge = self.numVehicles / self.edgeCount
        separation = edgeLength / carsPerEdge
        self.vehicles = []
        self.autoVehicles = []
                
        for j in range(self.edgeCount):
            for i in range(round(carsPerEdge)):
                    v = veh.Vehicle(veh.IDMController(**IDMControllerParams), None, self.saveSpeeds)
                    self.vehicles.append(v)
                    traci.vehicle.add(v.id, 'r'+str(j), departPos=i*separation)
                    traci.vehicle.setSpeedMode(v.id, 0)
        
        if self.selfDrivingPercentage > 0.0001:
            for v in self.vehicles[::int(1/self.selfDrivingPercentage)]:
                v.controller = veh.ObedientController()
                self.autoVehicles.append(v)
                traci.vehicle.setColor(v.id, (20,40,230))
                
    def __CalculateReward1(self, states, statesAuto, collisions):
        desSpeed = 11
        reward = sum(desSpeed - (np.abs(np.array([s.speed for s in statesAuto]) - desSpeed))) + np.mean([s.speed for s in states])
        if collisions:
            reward += self.accidentPunishment
            
        return reward
    
    def __CalculateReward2(self, states, statesAuto, newStates, newStatesAuto, collisions):
        desSpeed = 11
        reward = 5 - (5*sum([s.acceleration for s in states if s.acceleration < 0]) - sum([abs(s.speed-desSpeed) for s in states]))/(len(self.vehicles)*10)
        
        if collisions:
            reward += self.accidentPunishment
            
        return reward
    def __CalculateReward3(self, states, statesAuto, newStates, newStatesAuto, collisions):
        desSpeed = 11
        reward = 5 - (5*sum([s.acceleration for s in newStates if s.acceleration < 0]) - sum([abs(s.speed-desSpeed) for s in newStates]))/(len(self.vehicles)*20)

        reward -= sum([abs(s.speed-desSpeed) for s in newStatesAuto])
        
        if collisions:
            reward += self.accidentPunishment
                
        return reward
