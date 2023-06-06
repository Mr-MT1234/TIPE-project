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
INTERSECTION_WHITE_LINE = 0.5 
SWITCH_DELAY = 1

ROAD_WIDTH_PIX = 30
CAR_SIZE_PIX = 10

CAR_COLOR = (47,215,180)
AUTO_CAR_COLOR = (175,240,0)
ROAD_COLOR = (100,100,100)
BACKGROUND_COLOR = (200,200,200)

LEFT_DIRECT = -1
RIGHT_DIRECT = 1
UP_DIRECT = 0

OBSERVATION_RES = 64

MAX_CARS_PER_ARM = 10

IDM_a0 = 5.0
IDM_b0 = 5.0
IDM_T0 = 1
IDM_s0 = 10.0

@dataclass
class Vehicle:
    history : list
    direction : int
    creationIter : int
    position : float = 0
    velocity : float = 0
    acceleration : float = 0
    timeInIntersection : float = 0
    visualID : int = 1
    
def calcIDMAccel(speed, desiredSpeed, leaderSpeed, distance):
    v =  speed
    dv = v - leaderSpeed
    v0 = desiredSpeed
    sn = IDM_s0 + max(0, v*IDM_T0 + (v*dv)/(2*np.sqrt(IDM_a0*IDM_b0)))
    a = IDM_a0*(1 - (v/v0)**4 - (sn/distance)**2)
    
    return a

def calcSteadyAccel(speed, desiredSpeed, leaderSpeed, distance, maxAccel, dt):
        speedGoal = desiredSpeed
        impact = distance / (desiredSpeed - leaderSpeed)
        if distance < 10: speedGoal = 0
        elif impact < (speed / maxAccel): speedGoal = min(leaderSpeed, desiredSpeed)

        a = np.clip(((speedGoal - speed)/dt), -maxAccel, maxAccel)
        return a

def calcAccel(speed, desiredSpeed, maxAccel, dt):
    return np.clip((desiredSpeed - speed) / dt, -maxAccel, maxAccel) 

class IntersectionEnv(gym.Env):
    def __init__(self,  
        armLength    = 100,
        bodyLength   = 100,
        spawnRate    = 0.5,
        autoCount    = 3,
        timeStep     = 1/20,
        maxSpeed     = 30,
        turningSpeed = 3,
        maxAccel     = 5,
        maxIter      = 1000
    ):
        
        self.armLength = armLength
        self.bodyLength = bodyLength
        self.spawnRate = spawnRate
        self.autoCount = autoCount
        self.timeStep = timeStep
        self.maxSpeed = maxSpeed
        self.turningSpeed = turningSpeed
        self.maxAccel = maxAccel
        self.maxIter = maxIter
        self.maxStopTime = maxSpeed / maxAccel
        self.autoVehicles = []
        
        self.scale = min((WINDOW_WIDTH - 2*PADDING) / (2*(armLength) + ROAD_WIDTH), (WINDOW_HEIGTH - 2*PADDING) / (bodyLength + ROAD_WIDTH))
        
        self.leftArm = deque()
        self.leftHand = deque()
        self.rightArm = deque()
        self.rightHand = deque()
        self.body = deque()
        self.leftReturn = deque()
        self.rightReturn = deque()
        self.intersection = deque()
        
        self.returnLength = np.sqrt(armLength**2 + bodyLength**2)
        returnAngle = np.arctan(bodyLength / armLength)
        self.roads = [(self.rightArm, armLength, 0), (self.leftArm, armLength, np.pi),(self.body,bodyLength,np.pi/2),
                      (self.rightReturn, self.returnLength, returnAngle + np.pi), (self.leftReturn,self.returnLength, -returnAngle),
                      (self.rightHand, armLength, 0), (self.leftHand, armLength, np.pi)]

        self.intersectionQueue = deque()
        
        self.turnTime = np.pi  * ROAD_WIDTH / (4*turningSpeed)
        
        self.currentIter = 0
        self.traversalCount = 1
        
        self.metadata['render_modes'] = {'human','none'}
        self.observation_space = gym.spaces.Box(
            low=np.zeros((1,8,OBSERVATION_RES,2)),
            high=np.array([[[[200, maxSpeed]]*OBSERVATION_RES]*8]),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0,-1]*autoCount),
            high=np.array([maxSpeed,1]*autoCount),
            dtype=np.float32
        )
        
        pg.init()
        pg.font.init()
        self.font = pg.font.SysFont('arial.ttp',21)
        self.surface = pg.display.set_mode(size=(WINDOW_WIDTH,WINDOW_HEIGTH))
        pg.display.set_caption('T-juction Environment')
        self.isRenderReady = True
        
        self.reset()
        
        self.renderState = False
        self.lastAction = 0
        self.lastReward = 0
        self.lastState = []
    
    def render(self):
        for _ in pg.event.get():
            pass
        self.surface.fill(BACKGROUND_COLOR)
        if not self. renderState:
            
            centerX, centerY = WINDOW_WIDTH//2,WINDOW_HEIGTH//2
            
            bodyLengthPix = int(self.bodyLength*self.scale)
            armLengthPix  = int(self.armLength*self.scale)
            
            bodyRect = pg.Rect((centerX-ROAD_WIDTH_PIX//2, centerY - bodyLengthPix // 2), (ROAD_WIDTH_PIX, bodyLengthPix))
            leftArmRect = pg.Rect((centerX - ROAD_WIDTH_PIX / 2 - armLengthPix, centerY + bodyLengthPix // 2), (armLengthPix, ROAD_WIDTH_PIX))
            rightArmRect = pg.Rect((centerX + ROAD_WIDTH_PIX/2, centerY + bodyLengthPix // 2), (armLengthPix, ROAD_WIDTH_PIX))
            intersectionRect = pg.Rect( (centerX - ROAD_WIDTH_PIX/2,  centerY + bodyLengthPix // 2), (ROAD_WIDTH_PIX, ROAD_WIDTH_PIX) )
            
            sinInv = np.sqrt(self.bodyLength**2 + self.armLength**2) / self.bodyLength
            cosInv = np.sqrt(self.bodyLength**2 + self.armLength**2) / self.armLength
            
            returnPolyRight = np.array([(bodyRect.right, bodyRect.top + int(cosInv*ROAD_WIDTH_PIX)) , bodyRect.topright, rightArmRect.topright, (rightArmRect.right - int(sinInv*ROAD_WIDTH_PIX), rightArmRect.top) ])
            returnPolyLeft  = np.array([ (bodyRect.left, bodyRect.top + int(cosInv*ROAD_WIDTH_PIX)), bodyRect.topleft , leftArmRect.topleft, (leftArmRect.left + int(sinInv*ROAD_WIDTH_PIX), leftArmRect.top)])
            
            pg.draw.rect(self.surface, ROAD_COLOR, bodyRect)
            pg.draw.rect(self.surface, ROAD_COLOR, leftArmRect)
            pg.draw.rect(self.surface, ROAD_COLOR, rightArmRect)
            pg.draw.rect(self.surface, ROAD_COLOR, intersectionRect)
            pg.draw.polygon(self.surface, ROAD_COLOR, returnPolyRight)
            pg.draw.polygon(self.surface, ROAD_COLOR, returnPolyLeft)
            
            car = np.array([-1,0.5 -0.5j,0.5 + 0.5j], dtype=np.complex64) * CAR_SIZE_PIX
            
            leftArmPoints = np.array([ (leftArmRect.left,leftArmRect.centery), (leftArmRect.right,leftArmRect.centery) ])
            rightArmPoints = np.array([ (rightArmRect.right,rightArmRect.centery), (rightArmRect.left,rightArmRect.centery) ])
            bodyPoints = np.array([ (bodyRect.centerx,bodyRect.bottom), (bodyRect.centerx,bodyRect.top) ])
            rightReturnPoints = np.array([ np.average(returnPolyRight[:2], axis=0), np.average(returnPolyRight[2:], axis=0) ])
            leftReturnPoints  = np.array([ np.average(returnPolyLeft[:2], axis=0), np.average(returnPolyLeft[2:], axis=0) ])
            
            points = [rightArmPoints, leftArmPoints, bodyPoints, rightReturnPoints, leftReturnPoints]
            
            for (road, length, angle), ps in zip(self.roads, points):
                for v in road:
                    t = v.position / length
                    pos = ps[1]*t + (1-t)*ps[0]
                    pos = complex(pos[0], pos[1])
                    transformedCar = pos + np.exp(complex(0,angle)) * car
                    transformedCar = np.array( [(int(p.real), int(p.imag)) for p in transformedCar] )
                    
                    pg.draw.polygon(self.surface, CAR_COLOR if v.visualID == 1 else AUTO_CAR_COLOR, transformedCar)
                    
            for v in self.intersection:
                pos = complex(intersectionRect.centerx, intersectionRect.centery)
                t = v.timeInIntersection / self.turnTime
                
                if v.direction == LEFT_DIRECT:
                    transformedCar = -car
                    angle = (t - 1)*np.pi/2
                    pos = complex(leftArmRect.right, leftArmRect.top)
                else:
                    transformedCar = car
                    angle = (1-t)*np.pi/2
                    pos = complex(rightArmRect.left, rightArmRect.top)
                
                transformedCar = pos + np.exp(complex(0,angle)) * (transformedCar + complex(0,ROAD_WIDTH_PIX / 2))
                transformedCar = np.array( [(int(p.real), int(p.imag)) for p in transformedCar] )
                pg.draw.polygon(self.surface, CAR_COLOR if v.visualID == 1 else AUTO_CAR_COLOR, transformedCar)

            actionText = 'action: ' + str(self.lastAction)

            rewardText  = 'reward: %.2f' % (self.lastReward*100) + 'e-2'
            actionTextImage = self.font.render(actionText, True, (0,0,0))
            rewardTextImage = self.font.render(rewardText, True, (0,0,0))
            actionTextRect = actionTextImage.get_rect()
            actionTextRect.topleft = (10,10)
            self.surface.blit(actionTextImage, actionTextRect)
            rewardTextRect = rewardTextImage.get_rect()
            rewardTextRect.topleft = (10,30)
            self.surface.blit(rewardTextImage, rewardTextRect)
            
        else:
            stride = WINDOW_WIDTH / OBSERVATION_RES

            for i in range(len(self.lastState[0])):
                for j in range(len(self.lastState[0,i])):
                    color = (0,0,0)
                    if self.lastState[0,i,j,0] == 1 : color = CAR_COLOR
                    elif self.lastState[0,i,j,0] > 1 : color = AUTO_CAR_COLOR
                    pg.draw.rect(self.surface, color, pg.Rect((j*stride,i*stride),(stride,stride)))
            
        
        pg.display.flip()        
        
    def step(self, action):
        self.lastAction = action
        reward = 0
        
        if np.random.random() < self.spawnRate*self.timeStep:
            v = Vehicle(history=[], direction=0,visualID=1,creationIter=self.currentIter)
            s = np.random.random()
            if s > 0.5:
                if (not self.rightHand or self.rightHand[-1].position > 5) and len(self.rightArm) < MAX_CARS_PER_ARM:
                    v.direction = RIGHT_DIRECT
                    self.rightHand.append(v)
            else:
                if (not self.leftHand or self.leftHand[-1].position > 5) and len(self.leftArm) < MAX_CARS_PER_ARM:
                    v.direction = LEFT_DIRECT
                    self.leftHand.append(v)
                    
        for r, length, _ in self.roads:
            for v in r:
                v.position = min(v.position, length+0.5)
        
        for r in [self.rightArm, self.leftArm]:
            for v in r:
                b = RIGHT_DIRECT if r == self.rightArm else LEFT_DIRECT
                if (v.position > self.armLength - INTERSECTION_WHITE_LINE) and ((v,b) not in self.intersectionQueue):
                    self.intersectionQueue.append((v,b))
        
        if self.body and self.body[0].position > self.bodyLength:
            v = self.body.popleft()
            if v.visualID > 1:
                i = v.visualID - 2
                v.position = 0
                if action[2*i+1] > 0: 
                    v.direction = LEFT_DIRECT
                    self.rightReturn.append(v)
                else:
                    v.direction = RIGHT_DIRECT
                    self.leftReturn.append(v)
                
        if self.leftHand and self.leftHand[0].position > self.bodyLength and (not self.leftArm or self.leftArm[-1].position > 2):
            v = self.leftHand.popleft()
            v.position = 0
            self.leftArm.append(v)
        if self.rightHand and self.rightHand[0].position > self.bodyLength and (not self.rightArm or self.rightArm[-1].position > 2):
            v = self.rightHand.popleft()
            v.position = 0
            self.rightArm.append(v)
        
        if self.rightReturn and self.rightReturn[0].position > self.returnLength:
            v = self.rightReturn.popleft()
            v.position = 0
            self.rightArm.append(v)
        
        if self.leftReturn and self.leftReturn[0].position > self.returnLength:
            v = self.leftReturn.popleft()
            v.position = 0
            self.leftArm.append(v)
        
        for v in self.intersection:
            v.timeInIntersection -= self.timeStep
        
        if self.intersection and self.intersection[0].timeInIntersection < 0:
            v = self.intersection.popleft()
            v.direction = UP_DIRECT
            v.position = 0
            self.body.append(v)
            
        if self.intersectionQueue and not self.intersection and (not self.body or self.body[-1].position > 2):
            next, current = self.intersectionQueue.popleft()
            next = self.rightArm.popleft() if current == RIGHT_DIRECT else self.leftArm.popleft()
            next.timeInIntersection = self.turnTime
            
            if current/abs(current) != self.traversalCount / abs(self.traversalCount):
                next.timeInIntersection += SWITCH_DELAY
                self.traversalCount = current
            else:
                self.traversalCount += current
                reward += min(7,abs(self.traversalCount))*100
                
            self.intersection.append(next)
            self.lastIntersectionTraversal = current
            
        if self.rightArm:
            leader = self.rightArm[0]
            if leader.visualID > 1: 
                desiredSpeed = action[2*(leader.visualID-2) + 0]
            else:
                desiredSpeed = self.maxSpeed
            distanceToEnd = self.armLength - leader.position 
                
            if distanceToEnd < leader.velocity**2 / (2*self.maxAccel):
                shouldStop = bool(self.intersection) or \
                (bool(self.leftArm) and self.armLength - self.leftArm[0].position < distanceToEnd + 10)
                
                desiredSpeed = self.turningSpeed*(1-shouldStop)
            leader.acceleration = calcAccel(leader.velocity, desiredSpeed, self.maxAccel, self.timeStep)
        
        if self.leftArm:
            leader = self.leftArm[0]
            
            if leader.visualID > 1: 
                desiredSpeed = action[2*(leader.visualID-2) + 0]
            else:
                desiredSpeed = self.maxSpeed
            
            distanceToEnd = self.armLength - leader.position 
            if distanceToEnd <  leader.velocity**2 / (2*self.maxAccel):
                shouldStop = bool(self.intersection) or \
                (bool(self.rightArm) and self.armLength - self.rightArm[0].position < distanceToEnd + 10)
                
                desiredSpeed = self.turningSpeed*(1-shouldStop)
            leader.acceleration = calcAccel(leader.velocity, desiredSpeed, self.maxAccel, self.timeStep)
            
        
        for road, _, __ in self.roads:    
            for i in range(1, len(road)):
                v, leader = road[i], road[i-1]
                if v.visualID == 1:
                    v.acceleration = calcIDMAccel(v.velocity, self.maxSpeed, leader.velocity, max(0.01,leader.position - v.position))
                else:
                    i = v.visualID - 2
                    v.acceleration = calcSteadyAccel(v.velocity, action[2*i], leader.velocity,leader.position - v.position, self.maxAccel, self.timeStep)
        
        for road in [self.body, self.leftHand,self.rightHand]:
            if road:
                v = road[0]
                v.acceleration = calcAccel(road[0].velocity, self.maxSpeed, self.maxAccel, self.timeStep)
                
        for road in [self.leftReturn, self.rightReturn]:
            if road:
                v = road[0]
                i = v.visualID - 2
                v.acceleration = calcSteadyAccel(v.velocity, action[2*i], self.turningSpeed,self.returnLength + 11 - v.position, self.maxAccel, self.timeStep)
                            
        for r, _, __ in self.roads:
            for v in r:
                v.velocity = max(0,v.velocity + v.acceleration*self.timeStep)
                v.position += v.velocity*self.timeStep
        
        self.currentIter += 1
        
        # optSpeed = 25
        
        # vehicleCount = len(self.rightArm) + len(self.leftArm) + len(self.body) + \
        #             len(self.rightReturn) + len(self.leftReturn)
                    
        avg = 0
        for r, _, __ in self.roads:
            for v in r:
                reward -= 1
        
        if self.intersection:
            reward -= 1
                
        reward = reward / 100
        self.lastReward = reward
        
        
        state = self.getState()
        self.lastState = state
        return state, reward, self.currentIter >= self.maxIter, False, {}
    
    def reset(self):
        self.currentIter = 0
        
        self.leftArm.clear()
        self.rightArm.clear()
        self.rightHand.clear()
        self.rightHand.clear()
        self.body.clear()
        self.rightReturn.clear()
        self.leftReturn.clear()
        self.autoVehicles.clear()
        self.intersection.clear()
        self.intersectionQueue.clear()
        
        separation = 10
        
        sR = sL = 0
        
        for i in range(2,self.autoCount+2):
            v = Vehicle(history=[], visualID=i, direction=0,creationIter=0)
            r = np.random.random()
            if r < 0.5:
                v.direction = RIGHT_DIRECT
                v.position = sL
                sL += separation
                self.leftReturn.append(v)
            else:
                v.direction = LEFT_DIRECT
                v.position = sR
                sR += separation
                self.rightReturn.append(v)
            self.autoVehicles.append(v)
            
        self.leftReturn.reverse()
        self.rightReturn.reverse()
        
        return self.getState(), {}
    
    def close(self):
        pg.quit()
        
    def getState(self):
        state = np.zeros((1,8,OBSERVATION_RES,2), dtype=np.float32)
        for i, (road, length, _) in enumerate(self.roads):
            for v in road:
                t = int(v.position / length * OBSERVATION_RES)
                t = min(t, OBSERVATION_RES-1)
                state[0,i,t] = (v.visualID, v.velocity)
        
        intersectionData = (0,0)
        if self.intersection:
            intersectionData = (self.intersection[0].visualID, np.clip(self.traversalCount,-7,7))
        for i in range(OBSERVATION_RES):
            state[0,-1,i] = intersectionData
            
        return state