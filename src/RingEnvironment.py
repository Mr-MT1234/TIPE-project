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
__IDMControllerParams = {
    'a' :               5.0,
    'b' :               5.0,
    'T' :               2.0,
    'v' :               20.0,
    's' :               40.0,
    'imperfection' :    1/500
}

@dataclass
class __VehicleState:
    edgeID : str
    position : float
    speed : float
    leaderID : str
    leaderSpeed : float
    distanceToLeader : float

class RingEnv(gym.Env):
    
    def __init__(self, *, radius, numVehicles, selfDrivingPercentage = 0.05, timeStep = 0.5,saveSpeeds = False, renderMode = 'human'):
        super().__init__()
        
        assert renderMode in ['human', 'none'] 
        
        self.numVehicles = numVehicles
        self.radius = radius
        self.selfDrivingNum = int(selfDrivingPercentage * numVehicles)
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
                    
        for v in self.vehicles[::int(1/selfDrivingPercentage)]:
            v.controller = veh.ObedientController()
            self.autoVehicles.append(v)
                    
        
    def render(self):
        pass
    
    def step(self, action):
        
        observation = 0
        reward = 0
        terminated = 0
        truncated = 0
        info = {}
        done = False
    
    def reset(self):
        pass
    
    def close(self):
        pass
    
    def __FetchState(vehicle):
        edgeID = traci.vehicle.getRoadID(vehicle.id)
        leaderID, distance = traci.vehicle.getLeader(vehicle.id)
        speed = traci.vehicle.getSpeed(vehicle.id)
        leaderSpeed = traci.vehicle.getSpeed(vehicle.id) #leader
        position = traci.vehicle.getLanePosition(vehicle.id)
        
        if 
