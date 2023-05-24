from dataclasses import dataclass
import os
import sys 
from collections import deque

import gymnasium as gym
import numpy as np
import pygame as pg


PADDING = 30
WINDOW_WIDTH, WINDOW_HEIGTH = (700,700) 
ROAD_WIDTH = 4
ROAD_WIDTH_PIX = 30
CAR_SIZE_PIX = 10

CAR_COLOR = (99,65,229)
AUTO_CAR_COLOR = (229,99,65)
BACKGROUND_COLOR = (200,200,200)
ROAD_COLOR = (150,150,150)

@dataclass
class Vehicle:
    position : float
    velocity : float
    acceleration : float
    rotation : float
    history : list
    

class IntersectionEnv(gym.Env):
    def __init__(self,  armLength    = 100,
                        bodyLength   = 100,
                        spawnRate    = 0.5,
                        autoCount    = 3,
                        timeStep     = 1/20,
                        maxSpeed     = 30,
                        turningSpeed = 3,
                        maxAccel     = 3,
                        maxIter      = 1000):
        
        self.armLength = armLength
        self.bodyLength = bodyLength
        self.spawnRate = spawnRate
        self.autoCount = autoCount
        self.timeStep = timeStep
        self.maxSpeed = maxSpeed
        self.turningSpeed = turningSpeed
        self.maxAccel = maxAccel
        self.maxIter = maxIter
        
        self.scale = min((WINDOW_WIDTH - 2*PADDING) / (2*(armLength) + ROAD_WIDTH), (WINDOW_HEIGTH - 2*PADDING) / (bodyLength + ROAD_WIDTH))
        
        self.leftArm = deque()
        self.rightArm = deque()
        self.body = deque()
        self.leftReturn = deque()
        self.rightReturn = deque()
        self.intersection = deque()
        
        self.turnTime = np.pi  * ROAD_WIDTH / (4*turningSpeed)
        
        self.currentIter = 0
        
        pg.init()
        pg.font.init()
        self.font = pg.font.SysFont('arial.ttp',21)
        self.surface = pg.display.set_mode(size=(WINDOW_WIDTH,WINDOW_HEIGTH))
        pg.display.set_caption('T-juction Environment')
        self.isRenderReady = True
        
        pass
    
    def render(self):            
        for _ in pg.event.get():
            pass
        
        self.surface.fill(BACKGROUND_COLOR)
        
        centerX, centerY = WINDOW_WIDTH//2,WINDOW_HEIGTH//2
        
        bodyLengthPix = int(self.bodyLength*self.scale)
        armLengthPix  = int(self.armLength*self.scale)
        
        bodyRect = pg.Rect((centerX-ROAD_WIDTH_PIX//2, centerY - bodyLengthPix // 2), (ROAD_WIDTH_PIX, bodyLengthPix))
        leftArmRect = pg.Rect((centerX - ROAD_WIDTH_PIX / 2 - armLengthPix, centerY + bodyLengthPix // 2), (armLengthPix, ROAD_WIDTH_PIX))
        rightArmRect = pg.Rect((centerX + ROAD_WIDTH_PIX/2, centerY + bodyLengthPix // 2), (armLengthPix, ROAD_WIDTH_PIX))
        intersectionRect = pg.Rect( (centerX - ROAD_WIDTH_PIX/2,  centerY + bodyLengthPix // 2), (ROAD_WIDTH_PIX, ROAD_WIDTH_PIX) )
        
        sin = np.sqrt(self.bodyLength**2 + self.armLength**2) / self.bodyLength
        cos = np.sqrt(self.bodyLength**2 + self.armLength**2) / self.armLength
        
        returnPolyRight = np.array([ bodyRect.topright, rightArmRect.topright, (rightArmRect.right - int(sin*ROAD_WIDTH_PIX), rightArmRect.top), (bodyRect.right, bodyRect.top + int(cos*ROAD_WIDTH_PIX)) ])
        returnPolyLeft  = np.array([ bodyRect.topleft, leftArmRect.topleft, (leftArmRect.left + int(sin*ROAD_WIDTH_PIX), leftArmRect.top), (bodyRect.left, bodyRect.top + int(cos*ROAD_WIDTH_PIX)) ])
        
        pg.draw.rect(self.surface, ROAD_COLOR, bodyRect)
        pg.draw.rect(self.surface, ROAD_COLOR, leftArmRect)
        pg.draw.rect(self.surface, ROAD_COLOR, rightArmRect)
        pg.draw.rect(self.surface, ROAD_COLOR, intersectionRect)
        pg.draw.polygon(self.surface, ROAD_COLOR, returnPolyRight)
        pg.draw.polygon(self.surface, ROAD_COLOR, returnPolyLeft)
        
        
        car = np.array([1j,-0.5 -0.5j,0.5 -0.5j], dtype=np.complex64) * CAR_SIZE_PIX
        
        
        
        
        pg.display.flip()        
        
    def step(self, action):
        return None, 0, False, False, {}
    
    def reset(self):
        return None, {}
    
    def close():
        pg.quit()
        