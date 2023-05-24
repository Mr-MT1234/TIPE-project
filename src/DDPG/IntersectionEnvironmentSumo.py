import os
import sys 
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import traci
import sumolib

import Vehicle
from GenerateMaps import createIntersectionFiles

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

OBSERVATION_RESOLUTION = 300

    
@dataclass
class _VehicleData:
    id : str
    visualID : int
    edge : sumolib.net.edge.Edge
    controller : Vehicle.Controller
    position : float = 0
    speed : float = 0
    acceleration  : float = 0
    leaderSpeed : float = 0
    distance : float = 0
    cumilativeAccel : float = 0
    cumilativeWaitTime : float = 0
    
class IntersectionEnv(gym.Env):
    def __init__(self,
                 sumoFiles,
                 vehicleRate,
                 autoCount,
                 maxAccel = 2,
                 maxSpeed = 20,
                 timeStep = 0.5,
                 maxIter = 1000,
                 IDMControllerParams = IDM_CONTROLLER_DEFAULT_PARAMS,
                 renderMode = 'human'):
        
        self.vehicleRate = vehicleRate
        self.autoCount = autoCount
        self.maxAccel  = maxAccel
        self.maxSpeed  = maxSpeed
        self.timeStep  = timeStep
        self.maxIter   = maxIter
        self.sumoFiles = sumoFiles
        self.render_mode = renderMode
        self.IDMControllerParams = IDMControllerParams
        self.currentIter = 0
        self.vehicles = []
        self.lastWaitTime = 0
        
        createIntersectionFiles(200,300, sumoFiles)
        
        self.net = sumolib.net.readNet(sumoFiles + '/intersection.net.xml')
        
        sumoBinary = 'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui'
        if renderMode != 'human': 
            sumoBinary = 'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo'
        
        sumoCmd = [sumoBinary, '--step-length='+str(timeStep),'--no-warnings', '--no-step-log', "-c", sumoFiles + '/intersection.sumocfg']
         
        traci.start(sumoCmd)
        
        traci.route.add('rR', ['ePR'])
        traci.route.add('rL', ['ePL'])
        
        displacementR = displacementL = 0
        carLenght = 10
        
        for i in range(autoCount):
            v = _VehicleData(
                id=f'v{i}',
                edge = None,
                controller=Vehicle.ObedientController(self.maxAccel),
                visualID=10+i
            )
            if np.random.random() < 0.5:
                v.edge = self.net.getEdge('ePR')
                v.position = displacementR
                displacementR += carLenght
                routeID = 'rR'
            else:
                v.edge = self.net.getEdge('ePL')
                v.position = displacementL
                displacementL += carLenght
                routeID = 'rL'
                
                
            self.vehicles.append(v)
            
            traci.vehicle.add(v.id, routeID=routeID, departPos=v.position)
            traci.vehicle.setColor(v.id, (30, 150, 240))
            #traci.vehicle.setSpeedMode(v.id, 0)
            
        self.action_space = gym.spaces.Box(
            low=np.array([0, -1] * autoCount, dtype=np.float32),
            high=np.array([maxSpeed, 1] * autoCount, dtype=np.float32)
        )
                
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(5,OBSERVATION_RESOLUTION,2), dtype=np.float32
        )
        
        traci.route.add('LeftRoute', ['eLL','eLC','eCP', 'ePP'])
        traci.route.add('RightRoute', ['eRR','eRC','eCP', 'ePP'])
        
    def render(self):
        pass
    
    def step(self, action):
        directions = action.reshape((self.autoCount,2))[:,1]
        speeds = action.reshape((self.autoCount,2))[:,0]
        
        directions = [ 'ePR' if d > 0 else 'ePL' for d in directions]
        
        for auto, d, s in zip(self.vehicles[:self.autoCount], directions, speeds):
            if auto.edge.getID() == 'eCP':
                traci.vehicle.changeTarget(auto.id, d)
                
            auto.controller.setDesiredSpeed(s)
            
            auto.acceleration = auto.controller.calcAcceleration(self.timeStep, auto.speed, auto.leaderSpeed, auto.distance)
            
        for auto, s in zip(self.vehicles[:self.autoCount], speeds):
            traci.vehicle.setSpeed(auto.id, s + auto.acceleration*self.timeStep)
            
        traci.simulationStep()
            
        for i in range(len(self.vehicles) -1, -1, -1):
            if self.vehicles[i].id not in traci.vehicle.getIDList():
                del self.vehicles[i]
        
        for v in self.vehicles:
            self._UpdateVehiculeData(v)
            
        state = np.zeros((5,OBSERVATION_RESOLUTION,2), dtype=np.float32)
        for v in self.vehicles:
            relativePos = v.position / v.edge.getLength()  
            i = np.clip(int(relativePos * OBSERVATION_RESOLUTION),0,OBSERVATION_RESOLUTION-1)
            
            if v.edge.getID() == 'eLC':
                state[0][i] = (v.visualID, v.speed)
            elif v.edge.getID() == 'eRC':
                state[1][i] = (v.visualID, v.speed)
            elif v.edge.getID() == 'eCP':
                state[2][i] = (v.visualID, v.speed)
            elif v.edge.getID() == 'ePR':
                state[3][i] = (v.visualID, v.speed)
            elif v.edge.getID() == 'ePL':
                state[4][i] = (v.visualID, v.speed)
                
        #Spawning cars
        
        r = np.random.random()
        if r < self.vehicleRate:
            spawnSide = np.random.choice(2)
            v = _VehicleData(
                id=f'v{1000 + self.currentIter}',
                edge = self.net.getEdge('ePR' if spawnSide else 'ePL'),
                controller=Vehicle.ObedientController(self.maxAccel),
                visualID=1
            )
            traci.vehicle.add(v.id, routeID='RightRoute' if spawnSide else 'LeftRoute')
            self.vehicles.append(v)
           
        waitTime = traci.edge.getWaitingTime('eLC')
        reward = +self.lastWaitTime - waitTime
        
        self.lastWaitTime = waitTime
                
        truncated = False
        terminated = self.currentIter > self.maxIter
        self.currentIter += 1
                
        return state, reward, terminated, truncated, {}
                
    
    def _UpdateVehiculeData(self, v):
        eID = traci.vehicle.getRoadID(v.id)
        v.position = traci.vehicle.getLanePosition(v.id) 
        
        edgeCases = {
            ':nCenter_0': 'eRC',
            ':nCenter_1' : 'eLC',
            ':nCenter_2' : 'eLC',
            ':nPost_0' : 'eCP',
            ':nPost_1' : 'eCP',
            ':nPost_2' : 'eCP',
            ':nRight_0' : 'ePR',
            ':nRight_1' : 'eRC',
            ':nLeft_0' : 'ePL',
            ':nLeft_1' : 'eLL',
            ':nLeft_2' : 'ePL',
        }
        
        if eID in edgeCases:
            eID = edgeCases[eID]
            v.position = self.net.getEdge(eID).getLength()
        
        v.edge = self.net.getEdge(eID)
        v.speed= traci.vehicle.getSpeed(v.id)
        v.cumilativeAccel += abs(v.acceleration)
        v.cumilativeWaitTime = 0
        
        leader= traci.vehicle.getLeader(v.id, dist=2000)
        
        if leader:
            leaderID, distance = leader
            v.leaderSpeed = traci.vehicle.getSpeed(leaderID)
            v.distance = distance
        else:
            v.leaderSpeed = v.speed
            v.distance = np.inf
            
    def reset(self):
        for v in self.vehicles:
            traci.vehicle.remove(v.id)
        
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        
        Vehicle.ResetIDCounter()
        self.vehicles.clear()
            
        displacementR = displacementL = 0
        carLenght = 10
        
        
        for i in range(self.autoCount):
            v = _VehicleData(
                id=f'v{i}',
                edge = None,
                controller=Vehicle.ObedientController(self.maxAccel),
                visualID=10+i
            )
            if np.random.random() < 0.5:
                v.edge = self.net.getEdge('ePR')
                v.position = displacementR
                displacementR += carLenght
                routeID = 'rR'
            else:
                v.edge = self.net.getEdge('ePL')
                v.position = displacementL
                displacementL += carLenght
                routeID = 'rL'
                
            self.vehicles.append(v)
            
            traci.vehicle.add(v.id, routeID=routeID, departPos=v.position)
            traci.vehicle.setColor(v.id, (30, 150, 240))
            #traci.vehicle.setSpeedMode(v.id, 0)
            
        state = np.zeros((5,OBSERVATION_RESOLUTION,2), dtype=np.float32)
        for v in self.vehicles:
            relativePos = v.position / v.edge.getLength()  
            i = np.clip(int(relativePos * OBSERVATION_RESOLUTION),0,OBSERVATION_RESOLUTION-1)
            if v.edge.getID() == 'eLC':
                state[0][i] = (v.visualID, v.speed)
            elif v.edge.getID() == 'eRC':
                state[1][i] = (v.visualID, v.speed)
            elif v.edge.getID() == 'eCP':
                state[2][i] = (v.visualID, v.speed)
            elif v.edge.getID() == 'ePR':
                state[3][i] = (v.visualID, v.speed)
            elif v.edge.getID() == 'ePL':
                state[4][i] = (v.visualID, v.speed)
                
        self.currentIter = 0
            
        return state, {}
        
