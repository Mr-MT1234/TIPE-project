import os
import sys 
import threading
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pygame as pg

import Vehicle
from GenerateMaps import createRingFiles


tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

IDM_CONTROLLER_DEFAULT_PARAMS = {
    'a' :               5.0,
    'b' :               5.0,
    'T' :               1,
    'v' :               15.0,
    's' :               3.0,
    'imperfection' :    1/500
}
ROAD_WIDTH = 50
BACKGROUND_COLOR = (200,200,200)
CAR_SIZE = 10
CAR_COLOR = (47,215,180)
AUTO_CAR_COLOR = (175,240,0)
ROAD_COLOR = (100,100,100)


@dataclass
class _VehicleData:
    position : float
    speed : float
    acceleration  : float
    distance : float
    controller : Vehicle.Controller
    history : list
    cumilativeAccel : float

class ConvRingEnv(gym.Env):
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
                 saveSpeeds = False):        
        super().__init__()
        
        assert renderMode in ['human', 'none'] 
        Vehicle.ResetIDCounter()
        self.radius = radius
        self.numVehicles = numVehicles
        self.numVehiclesAuto = numVehiclesAuto
        self.maxSpeed = maxSpeed
        self.maxAccel = maxAccel
        self.timeStep = timeStep
        self.maxIter = maxIter
        self.IDMControllerParams = IDMControllerParams
        self.saveSpeeds = saveSpeeds
        self.edgeCount = 4
        self.currentIter = 0
        self.currentVehicleState = None
        self.trackLenght = radius*2*np.pi
        self.lastAction = np.zeros((numVehiclesAuto,),dtype=np.float32)
        self.isRenderInit = (renderMode == 'human')
        
        self.render_mode = renderMode
        self.metadata['render_modes'] = {'human','none'}
        self.observation_space = gym.spaces.Box(
            low =np.zeros((numVehicles, 3)),
            high=np.array([[np.inf ,self.maxSpeed, 1]]*(numVehicles))
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0]*numVehiclesAuto),
            high=np.array([self.maxSpeed]*numVehiclesAuto)
        )
        
        self._PopulateRoad()
        
        if renderMode == 'human':
            self._InitRender()
            
    def render(self):
        
        if not self.isRenderInit:
            self._InitRender()
            
        for _ in pg.event.get():
            pass
        
        self.surface.fill(BACKGROUND_COLOR)
        
        centreX, centreY = self.surface.get_size()
        centre = (centreX//2,centreY//2)
        
        Ro = min(centreX,centreY) / 2
        Ri = Ro - ROAD_WIDTH
        R = Ro - ROAD_WIDTH / 2
        pg.draw.circle(self.surface, (150,150,150), centre, Ro)
        pg.draw.circle(self.surface, BACKGROUND_COLOR, centre, Ri)
        
        car = np.array([1j,-0.5 -0.5j,0.5 -0.5j], dtype=np.complex64) * CAR_SIZE
        
        points = [0]*3
        
        for v in self.vehicles:
            angle = v.position / self.trackLenght * (2*np.pi)
            w = np.exp(1j*angle)
            ps = (car * w + R*w) + centreX / 2 + (centreY / 2)*1j
            for i in range(3):
                points[i] = int(np.real(ps[i])), int(np.imag(ps[i]))
                
            color = CAR_COLOR if isinstance(v.controller,Vehicle.IDMController) else AUTO_CAR_COLOR
            pg.draw.polygon(self.surface, color, points)
            
        text = 'action: [' + '%.2f' % self.lastAction[0]
        for e in self.lastAction[1:]:
            text += '    %.2f' % e
        text += ']'
        textImage = self.font.render(text, True, (0,0,0))
        textRect = textImage.get_rect()
        textRect.topleft = (10,10)
        self.surface.blit(textImage, textRect)
        
        pg.display.flip()        
    
    def step(self, action):
        assert len(action)==0 or (([0]*self.numVehiclesAuto <= action).any() and (action <= [self.maxSpeed]*self.numVehiclesAuto).any()), 'Inacceptable action: out bounds'
        
        self.lastAction = action
        
        for vhID,v in zip(self.autoVehiclesIndeces, action):
            self.vehicles[vhID].controller.setDesiredSpeed(v)
        
        for i in range(self.numVehicles):
            v = self.vehicles[i]
            leader = self.vehicles[(i+1) % self.numVehicles]
            a = v.controller.calcAcceleration(self.timeStep, v.speed, leader.speed, v.distance)
            v.acceleration = a
            
        for v in self.vehicles:
            v.position += v.speed * self.timeStep
            v.speed = max(v.speed + v.acceleration * self.timeStep,0)
        
        for i in range(self.numVehicles - 1):
            v = self.vehicles[i]
            leader = self.vehicles[(i+1) % self.numVehicles]
            
            v.distance = leader.position - v.position
        
        self.vehicles[-1].distance = self.vehicles[0].position - self.vehicles[-1].position + self.trackLenght
        
        collision = False
        for v in self.vehicles:
            if v.distance <= 0:
                collision = True
                break
        
        state = np.empty((1,*self.observation_space.shape), dtype=np.float32)
        
        for i,v in enumerate(self.vehicles):
            state[0][i][0] = v.position
            state[0][i][1] = v.speed
            state[0][i][2] = 1 if isinstance(v.controller, Vehicle.ObedientController) else 0
        
        
        terminated = False
        truncated  = False
        reward = 0
        
        if collision:
            terminated = True
        
        if self.currentIter > self.maxIter:
            truncated = True
            
        desiredSpeed = 15
        hmax = 1
        positions = state[0,:,0]
        speeds = state[0,:,1]
        distances = np.array([v.distance for v in self.vehicles])
        accelerations = np.array([v.acceleration for v in self.vehicles])
        headways = distances[self.autoVehiclesIndeces]/np.maximum(speeds[self.autoVehiclesIndeces],0.01)
        
        
        reward = (desiredSpeed - abs(desiredSpeed - np.mean(speeds)) - 0.1*np.sum(np.maximum(hmax - headways,0)))*0.1
        
        
        self.currentIter += 1
        
        return state, reward, terminated, truncated, {}        
    
    def reset(self):
        self.currentIter = 0
        self._PopulateRoad()
        
        state = np.empty((1,*self.observation_space.shape), dtype=np.float32)
        
        for i,v in enumerate(self.vehicles):
            state[0][i][0] = v.position
            state[0][i][1] = v.speed
            state[0][i][2] = 1 if isinstance(v.controller, Vehicle.ObedientController) else 0
        
        return state, {}
    
    def close(self):
        pg.quit()
    
    def _PopulateRoad(self):
        separation = self.trackLenght / self.numVehicles
        self.vehicles = []
        self.autoVehiclesIndeces = []
                
        for i in range(self.numVehicles):
            pos = i*separation
            v = _VehicleData(position=pos,
                             speed=0,
                             acceleration=0,
                             distance = separation,
                             controller=Vehicle.IDMController(**self.IDMControllerParams))
            self.vehicles.append(v)
        
        firstAuto = np.random.randint(0, self.numVehicles)
        
        for i in range(self.numVehiclesAuto):
            j = int(i*(self.numVehicles/self.numVehiclesAuto) + firstAuto) % self.numVehicles
            v = self.vehicles[j]
            v.controller = Vehicle.ObedientController(self.maxAccel)
            self.autoVehiclesIndeces.append(j)
        self.autoVehiclesIndeces.sort()
        
    def _InitRender(self):
        pg.init()
        pg.font.init()
        self.font = pg.font.SysFont('arial.ttp',21)
        self.surface = pg.display.set_mode(size=(700,700))
        pg.display.set_caption('Ring Environment')
        
class RingEnv(ConvRingEnv):
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
                 saveSpeeds = False):
        super().__init__(radius, numVehicles, numVehiclesAuto, maxSpeed, maxAccel,
                         timeStep, maxIter, IDMControllerParams, renderMode, saveSpeeds)
        
    def step(self,action):
        newState, reward, terminated, truncated, info = super().step(action)
        return newState[0,:,:].reshape(-1), reward, terminated, truncated, info
    
    def reset(self):
        state, info = super().reset()
        return state[0,:,:].reshape(-1), info
        
class RingEnvSimple(gym.Env):
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
                 saveSpeeds = False):        
        super().__init__()
        
        assert renderMode in ['human', 'none'] 
        Vehicle.ResetIDCounter()
        self.radius = radius
        self.numVehicles = numVehicles
        self.numVehiclesAuto = numVehiclesAuto
        self.maxSpeed = maxSpeed
        self.maxAccel = maxAccel
        self.timeStep = timeStep
        self.maxIter = maxIter
        self.IDMControllerParams = IDMControllerParams
        self.saveSpeeds = saveSpeeds
        self.edgeCount = 4
        self.currentIter = 0
        self.currentVehicleState = None
        self.trackLenght = radius*2*np.pi
        self.isRenderReady = False
        self.lastAction = np.zeros((numVehiclesAuto,),dtype=np.float32)
        self.lastReward = 0
        
        self.metadata['render_modes'] = {'human','none'}
        self.observation_space = gym.spaces.Box(
            low =np.zeros((numVehiclesAuto*5,)),
            high=np.array([self.maxSpeed, np.inf , self.maxSpeed, np.inf , self.maxSpeed]*(numVehiclesAuto))
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0]*numVehiclesAuto),
            high=np.array([self.maxSpeed]*numVehiclesAuto)
        )
        
        self._PopulateRoad()
            
    def render(self):
        assert self.isRenderReady, 'Rendering should be initialised before calling the render function'
            
        for _ in pg.event.get():
            pass
        
        self.surface.fill(BACKGROUND_COLOR)
        
        centreX, centreY = self.surface.get_size()
        centre = (centreX//2,centreY//2)
        
        Ro = min(centreX,centreY) / 2
        Ri = Ro - ROAD_WIDTH
        R = Ro - ROAD_WIDTH / 2
        pg.draw.circle(self.surface, (150,150,150), centre, Ro)
        pg.draw.circle(self.surface, BACKGROUND_COLOR, centre, Ri)
        
        car = np.array([1j,-0.5 -0.5j,0.5 -0.5j], dtype=np.complex64) * CAR_SIZE
        
        points = [0]*3
        
        for v in self.vehicles:
            angle = v.position / self.trackLenght * (2*np.pi)
            w = np.exp(1j*angle)
            ps = (car * w + R*w) + centreX / 2 + (centreY / 2)*1j
            for i in range(3):
                points[i] = int(np.real(ps[i])), int(np.imag(ps[i]))
                
            color = CAR_COLOR if isinstance(v.controller,Vehicle.IDMController) else AUTO_CAR_COLOR
            pg.draw.polygon(self.surface, color, points)
            
        actionText = 'action: [' + '%.2f' % self.lastAction[0]
        for e in self.lastAction[1:]:
            actionText += '    %.2f' % e
        actionText += ']'
        rewardText  = 'reward: %.2f' % (self.lastReward*100) + 'e-2'
        actionTextImage = self.font.render(actionText, True, (0,0,0))
        rewardTextImage = self.font.render(rewardText, True, (0,0,0))
        actionTextRect = actionTextImage.get_rect()
        actionTextRect.topleft = (10,10)
        self.surface.blit(actionTextImage, actionTextRect)
        rewardTextRect = rewardTextImage.get_rect()
        rewardTextRect.topleft = (10,30)
        self.surface.blit(rewardTextImage, rewardTextRect)
        
        pg.display.flip()        
    
    def step(self, action):
        assert len(action)==0 or (([0]*self.numVehiclesAuto <= action).any() and (action <= [self.maxSpeed]*self.numVehiclesAuto).any()), 'Inacceptable action: out bounds'
        
        
        for vhID,v in zip(self.autoVehiclesIndeces, action):
            self.vehicles[vhID].controller.setDesiredSpeed(v)
        
        for i in range(self.numVehicles):
            v = self.vehicles[i]
            leader = self.vehicles[(i+1) % self.numVehicles]
            a = v.controller.calcAcceleration(self.timeStep, v.speed, leader.speed, v.distance)
            v.acceleration = a
            
        for v in self.vehicles:
            v.position += v.speed * self.timeStep
            v.speed = max(v.speed + v.acceleration * self.timeStep,0)
        
        for i in range(self.numVehicles - 1):
            v = self.vehicles[i]
            leader = self.vehicles[(i+1) % self.numVehicles]
            
            v.distance = leader.position - v.position
        
        self.vehicles[-1].distance = self.vehicles[0].position - self.vehicles[-1].position + self.trackLenght
        
        collision = False
        for v in self.vehicles:
            if v.distance <= 0:
                collision = True
                break
        
        speeds = np.array([v.speed for v in self.vehicles])
        distances = np.array([v.distance for v in self.vehicles])
        headways = distances[self.autoVehiclesIndeces]/np.maximum(speeds[self.autoVehiclesIndeces],0.01)

        state = np.empty((self.observation_space.shape[0]), dtype=np.float32)
        
        j = 0
        for i in self.autoVehiclesIndeces:
            state[5*j+0] = speeds[i]
            state[5*j+1] = distances[i]
            state[5*j+2] = speeds[(i+1) % self.numVehicles]
            state[5*j+3] = distances[i-1]
            state[5*j+4] = speeds[i-1]
            
            j += 1
        
        terminated = False
        truncated  = False
        reward = 0
        
        if collision:
            terminated = True
        
        if self.currentIter > self.maxIter:
            truncated = True
            
        desiredSpeed = 15
        hmax = 1
        
        reward = (desiredSpeed - abs(desiredSpeed - np.mean(speeds)) - 0.1*np.sum(np.maximum(hmax - headways,0)))*0.01
        
        self.currentIter += 1
        self.lastAction = action
        self.lastReward = reward
        
        return state, reward, terminated, truncated, {}        
    
    def reset(self):
        self.currentIter = 0
        self._PopulateRoad()
        
        speeds = np.array([v.distance for v in self.vehicles])
        distances = np.array([v.distance for v in self.vehicles])
        
        state = np.empty((self.observation_space.shape[0]), dtype=np.float32)
        
        j = 0
        for i in self.autoVehiclesIndeces:
            state[5*j+0] = speeds[i]
            state[5*j+1] = distances[i]
            state[5*j+2] = speeds[(i+1) % self.numVehicles]
            state[5*j+3] = distances[i-1]
            state[5*j+4] = speeds[i-1]
            
            j += 1
        
        return state, {}
    
    def close(self):
        pg.quit()
    
    def _PopulateRoad(self):
        separation = self.trackLenght / self.numVehicles
        self.vehicles = []
        self.autoVehiclesIndeces = []
                
        for i in range(self.numVehicles):
            pos = i*separation
            v = _VehicleData(position=pos,
                             speed=0,
                             acceleration=0,
                             distance = separation,
                             controller=Vehicle.IDMController(**self.IDMControllerParams))
            self.vehicles.append(v)
        
        firstAuto = np.random.randint(0, self.numVehicles)
        
        for i in range(self.numVehiclesAuto):
            j = int(i*(self.numVehicles/self.numVehiclesAuto) + firstAuto) % self.numVehicles
            v = self.vehicles[j]
            v.controller = Vehicle.ObedientController(self.maxAccel)
            self.autoVehiclesIndeces.append(j)
        self.autoVehiclesIndeces.sort()
        
    def InitRender(self):
        if not self.isRenderReady:
            pg.init()
            pg.font.init()
            self.font = pg.font.SysFont('arial.ttp',21)
            self.surface = pg.display.set_mode(size=(700,700))
            pg.display.set_caption('Ring Environment')
            self.isRenderReady = True
    
    def EndRender(self):
        if self.isRenderReady:
            pg.quit()
            self.isRenderReady = False
        
        
class RingEnvAILess(gym.Env):
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
                 saveStates = False):        
        super().__init__()
        
        assert renderMode in ['human', 'none'] 
        Vehicle.ResetIDCounter()
        self.radius = radius
        self.numVehicles = numVehicles
        self.numVehiclesAuto = numVehiclesAuto
        self.maxSpeed = maxSpeed
        self.maxAccel = maxAccel
        self.timeStep = timeStep
        self.maxIter = maxIter
        self.IDMControllerParams = IDMControllerParams
        self.saveStates = saveStates
        self.edgeCount = 4
        self.currentIter = 0
        self.currentVehicleState = None
        self.trackLenght = radius*2*np.pi
        self.isRenderReady = False
        self.lastAction = np.zeros((numVehiclesAuto,),dtype=np.float32)
        self.lastReward = 0
        
        self.metadata['render_modes'] = {'human','none'}
        self.observation_space = gym.spaces.Box(
            low =np.zeros((numVehiclesAuto*5,)),
            high=np.array([self.maxSpeed, np.inf , self.maxSpeed, np.inf , self.maxSpeed]*(numVehiclesAuto))
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0]*numVehiclesAuto),
            high=np.array([self.maxSpeed]*numVehiclesAuto)
        )
        
        self._PopulateRoad()
            
    def render(self):
        assert self.isRenderReady, 'Rendering should be initialised before calling the render function'
            
        for _ in pg.event.get():
            pass
        
        self.surface.fill(BACKGROUND_COLOR)
        
        centreX, centreY = self.surface.get_size()
        centre = (centreX//2,centreY//2)
        
        Ro = min(centreX,centreY) / 2
        Ri = Ro - ROAD_WIDTH
        R = Ro - ROAD_WIDTH / 2
        pg.draw.circle(self.surface, ROAD_COLOR, centre, Ro)
        pg.draw.circle(self.surface, BACKGROUND_COLOR, centre, Ri)
        
        car = np.array([1j,-0.5 -0.5j,0.5 -0.5j], dtype=np.complex64) * CAR_SIZE
        
        points = [0]*3
        
        for v in self.vehicles:
            angle = v.position / self.trackLenght * (2*np.pi)
            w = np.exp(1j*angle)
            ps = (car * w + R*w) + centreX / 2 + (centreY / 2)*1j
            for i in range(3):
                points[i] = int(np.real(ps[i])), int(np.imag(ps[i]))
                
            color = CAR_COLOR if isinstance(v.controller,Vehicle.IDMController) else AUTO_CAR_COLOR
            pg.draw.polygon(self.surface, color, points)
            
        actionText = 'action: ['
        if self.lastAction:
            actionText += + '%.2f' % self.lastAction[0]
        for e in self.lastAction[1:]:
            actionText += '    %.2f' % e
        actionText += ']'
        rewardText  = 'reward: %.2f' % (self.lastReward*100) + 'e-2'
        actionTextImage = self.font.render(actionText, True, (0,0,0))
        rewardTextImage = self.font.render(rewardText, True, (0,0,0))
        actionTextRect = actionTextImage.get_rect()
        actionTextRect.topleft = (10,10)
        self.surface.blit(actionTextImage, actionTextRect)
        rewardTextRect = rewardTextImage.get_rect()
        rewardTextRect.topleft = (10,30)
        self.surface.blit(rewardTextImage, rewardTextRect)
        
        pg.display.flip()        
    
    def step(self, action):
        assert len(action)==0 or (([0]*self.numVehiclesAuto <= action).any() and (action <= [self.maxSpeed]*self.numVehiclesAuto).any()), 'Inacceptable action: out bounds'
        
        
        # for vhID,v in zip(self.autoVehiclesIndeces, action):
        #     self.vehicles[vhID].controller.setDesiredSpeed(v)
        
        for i in range(self.numVehicles):
            v = self.vehicles[i]
            leader = self.vehicles[(i+1) % self.numVehicles]
            follower = self.vehicles[(i-1) % self.numVehicles]
            a = v.controller.calcAcceleration(self.timeStep, v.speed, leader.speed, v.distance, follower.speed, follower.distance)
            v.acceleration = a
            v.cumilativeAccel += abs(a)
            
        for v in self.vehicles:
            v.position += v.speed * self.timeStep
            v.speed = max(v.speed + v.acceleration * self.timeStep,0)

        if self.saveStates:
            for v in self.vehicles:
                v.history.append((v.position,v.speed))
        
        for i in range(self.numVehicles - 1):
            v = self.vehicles[i]
            leader = self.vehicles[(i+1) % self.numVehicles]
            
            v.distance = leader.position - v.position
        
        self.vehicles[-1].distance = self.vehicles[0].position - self.vehicles[-1].position + self.trackLenght
        
        collision = False
        for v in self.vehicles:
            if v.distance <= 0:
                collision = True
                break
        
        speeds = np.array([v.speed for v in self.vehicles])
        distances = np.array([v.distance for v in self.vehicles])
        headways = distances[self.autoVehiclesIndeces]/np.maximum(speeds[self.autoVehiclesIndeces],0.01)

        state = np.empty((self.observation_space.shape[0]), dtype=np.float32)
        
        j = 0
        for i in self.autoVehiclesIndeces:
            state[5*j+0] = speeds[i]
            state[5*j+1] = distances[i]
            state[5*j+2] = speeds[(i+1) % self.numVehicles]
            state[5*j+3] = distances[i-1]
            state[5*j+4] = speeds[i-1]
            
            j += 1
        
        terminated = False
        truncated  = False
        reward = 0
        
        if collision:
            terminated = True
        
        if self.currentIter > self.maxIter:
            truncated = True
            
        desiredSpeed = 15
        hmax = 1
        
        reward = (desiredSpeed - abs(desiredSpeed - np.mean(speeds)) - 0.1*np.sum(np.maximum(hmax - headways,0)))*0.01
        
        self.currentIter += 1
        self.lastAction = action
        self.lastReward = reward
        
        return state, reward, terminated, truncated, {}        
    
    def reset(self):
        self.currentIter = 0
        self._PopulateRoad()
        
        speeds = np.array([v.speed for v in self.vehicles])
        distances = np.array([v.distance for v in self.vehicles])
        
        state = np.empty((self.observation_space.shape[0]), dtype=np.float32)
        
        j = 0
        for i in self.autoVehiclesIndeces:
            state[5*j+0] = speeds[i]
            state[5*j+1] = distances[i]
            state[5*j+2] = speeds[(i+1) % self.numVehicles]
            state[5*j+3] = distances[i-1]
            state[5*j+4] = speeds[i-1]
            
            j += 1
            
        
        return state, {}
    
    def close(self):
        pg.quit()
    
    def _PopulateRoad(self):
        separation = self.trackLenght / self.numVehicles
        self.vehicles = []
        self.autoVehiclesIndeces = []
                
        for i in range(self.numVehicles):
            pos = i*separation
            v = _VehicleData(position=pos,
                             speed=0,
                             acceleration=0,
                             distance = separation,
                             controller=Vehicle.IDMController(**self.IDMControllerParams),
                             history=[],
                             cumilativeAccel=0)
            self.vehicles.append(v)
        
        firstAuto = np.random.randint(0, self.numVehicles)
        
        for i in range(self.numVehiclesAuto):
            j = int(i*(self.numVehicles/self.numVehiclesAuto) + firstAuto) % self.numVehicles
            v = self.vehicles[j]
            v.controller = Vehicle.SteadyController(self.maxAccel,9)
            #v.controller = Vehicle.MiddleController(self.maxAccel)
            self.autoVehiclesIndeces.append(j)
        self.autoVehiclesIndeces.sort()
        
    def InitRender(self):
        if not self.isRenderReady:
            pg.init()
            pg.font.init()
            self.font = pg.font.SysFont('arial.ttp',21)
            self.surface = pg.display.set_mode(size=(700,700))
            pg.display.set_caption('Ring Environment')
            self.isRenderReady = True
    
    def EndRender(self):
        if self.isRenderReady:
            pg.quit()
            self.isRenderReady = False
        