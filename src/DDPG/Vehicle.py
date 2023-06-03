import random
import numpy as np

class Controller:
    def calcAcceleration(self, dt, speed, distanceToLeader, followerSpeed, distanceToFollower):
        return 0
    def EmergencyBreak(self, breakTime):
        pass

_MaxImperfectionBreakTime = 3

class IDMController(Controller):

    def __init__(self, a, b, T , v, s, imperfection):
        self.a0 = a
        self.b0 = b
        self.T0 = T
        self.v0 = v
        self.s0 = s
        self.imperfection = imperfection
        
        self.breakTime = 0
    def calcAcceleration(self, dt, speed, leaderSpeed, distance, followerSpeed, distanceToFollower):
        r = random.random()
        if r < self.imperfection:
            self.EmergencyBreak(random.uniform(_MaxImperfectionBreakTime/2,_MaxImperfectionBreakTime))
        if self.breakTime > 0:
            self.breakTime -= dt
            return -self.b0
        
        if distance <= 0:
            return self.a0*(1-speed/self.v0)
        v =  speed
        dv = v - leaderSpeed
        sn = self.s0 + max(0, v*self.T0 + (v*dv)/(2*np.sqrt(self.a0*self.b0)))
        a = self.a0*(1 - (v/self.v0)**4 - (sn/distance)**2)
        return a
    def EmergencyBreak(self, breakTime):
        self.breakTime = breakTime
        
class ObedientController(Controller):
    def __init__(self, maxAccel):
        self.desSpeed = 0
        self.maxAccel = maxAccel
        
    def setDesiredSpeed(self, speed):
        self.desSpeed = speed
    
    def calcAcceleration(self, dt, speed, distanceToLeader, followerSpeed, distanceToFollower):
        return np.clip(((self.desSpeed - speed)/dt), -self.maxAccel, self.maxAccel)
    
    def EmergencyBreak(self, breakTime):
        pass
    
class MiddleController(Controller):
    def __init__(self, maxAccel):
        self.maxAccel = maxAccel

    def calcAcceleration(self, dt, speed, leaderSpeed, distanceToLeader, followerSpeed, distanceToFollower):
        goalSpeed = (distanceToFollower - distanceToLeader) / (2*dt) 
        return np.clip((speed - goalSpeed)/dt,-self.maxAccel, self.maxAccel)
    
    def EmergencyBreak(self, breakTime):
        pass
    
class SteadyController(Controller):
    def __init__(self, maxAccel, desSpeed):
        self.desSpeed = desSpeed
        self.maxAccel = maxAccel
        self.T = 0
        
    def setDesiredSpeed(self, speed):
        self.desSpeed = speed
    
    def calcAcceleration(self, dt, speed, leaderSpeed, distance, followerSpeed, distanceToFollower):
        speedGoal = self.desSpeed
        impact = distance / (speed - leaderSpeed)
        if impact < (speed / self.maxAccel): speedGoal = min(leaderSpeed, self.desSpeed)
        a = np.clip(((speedGoal - speed)/dt), -self.maxAccel, self.maxAccel)
        return a
    
    def EmergencyBreak(self, breakTime):
        pass

VID = 0
class Vehicle:
    def __init__(self, controller, router, saveSpeed = False):
        global VID
        self.id = 'v'+str(VID)
        VID += 1
        self.controller = controller
        self.router = router
        self.saveSpeed = saveSpeed
        self.posHistory = []
        self.a = 0
        self.v = 0
    
    def update(self,dt,state):
        a = self.controller.calcAcceleration(dt,self.v, state.leaderSpeed, state.distance)
        v = self.v + a*dt
        if self.saveSpeed:
            self.speedHistory.append(v)
        return v
    
def ResetIDCounter():
    global VID
    VID = 0
    
class Router:
    def getNextGoal(self, graph):
        return 0
    
class PerpetualRouter:
    def getNextGoal(self, edges):
        return np.random.choice(edges)