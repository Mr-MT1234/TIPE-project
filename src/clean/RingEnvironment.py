from dataclasses import dataclass

import numpy as np
import pygame as pg

import Vehicle

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

class RingEnv:
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
        self.numVehiclesAuto = numVehiclesAuto
        
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
            
        pg.display.flip()        
    
    def step(self):
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
        
        terminated = collision or self.currentIter > self.maxIter
        
        self.currentIter += 1
        
        return terminated
            
    def reset(self):
        self.currentIter = 0
        self._PopulateRoad()
        
    def close(self):
        pg.quit()
    
    def _PopulateRoad(self):
        separation = self.trackLenght / self.numVehicles
        self.vehicles = []
                
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
            
        i = np.random.randint(0,self.numVehicles)
        
        for k in range(self.numVehiclesAuto):
            v = self.vehicles[(i + k*(self.numVehicles // self.numVehiclesAuto))%self.numVehicles]
            v.controller = Vehicle.SteadyController(self.maxAccel, 7.5)
                
                
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
        