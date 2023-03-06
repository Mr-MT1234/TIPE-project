import random
import math

class Controller:
    def calcAcceleration(self, state):
        return 0
    def EmergencyBreak(self, breakTime):
        pass
class Router:
    def getNextGoal(self, state):
        return 0

__MaxImperfectionBreakTime = 2

class IDMController(Controller):

    def __init__(self, a, b, T , v, s, imperfection):
        self.a0 = a
        self.b0 = b
        self.T0 = T
        self.v0 = v
        self.s0 = s
        self.imperfection = imperfection
        
        self.breakTime = 0
    def calcAcceleration(self, dt, speed, leaderSpeed, distance):
        r = random.random()
        if r < self.imperfection:
            self.EmergencyBreak(random.uniform(__MaxImperfectionBreakTime/2,__MaxImperfectionBreakTime))
        if self.breakTime > 0:
            self.breakTime -= dt
            return -self.b0
        
        if distance < 0:
            return self.a0*(1-speed/self.v0)
        v =  speed
        dv = v - leaderSpeed
        sn = self.s0 + max(0, v*self.T0 + (v*dv)/(2*math.sqrt(self.a0*self.b0)))
        a = self.a0*(1 - (v/self.v0)**4 - (sn/distance)**2)
        return a
    def EmergencyBreak(self, breakTime):
        self.breakTime = breakTime
        
class __State:
    def __init__(self, speed = 0, leaderSpeed = '', distance = -1,dt = 0.1):
        self.leaderSpeed = leaderSpeed
        self.distance = distance
        
        
class ObedientController(Controller):

    def __init__(self):
        self.breakTime = 0
        self.a = 0
        
    def setNextAcceleration(self, a):
        self.a = a
    
    def calcAcceleration(self, dt, speed, leaderSpeed, distance):
        return self.a
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
        self.speedHistory = []
        self.a = 0
        self.v = 0
    
    def update(self,dt,state):
        a = self.controller.calcAcceleration(dt,self.v, state.leaderSpeed, state.distance)
        v = self.v + a*dt
        if self.saveSpeed:
            self.speedHistory.append(v)
        return v