import RingEnvironment
import keyboard
import time
import numpy as np

running = True

env = RingEnvironment.RingEnv(
                radius = 25,
                numVehicles = 10,
                selfDrivingPercentage = 1/20,
                timeStep = 0.5,
                saveSpeeds = False,
                selfDrivingAccel = 2,
                accidentPunishmeent = -1000,
                renderMode = 'human')

while running:
    if keyboard.is_pressed('e'):
        running = False
    
    a = 1
    if keyboard.is_pressed('j'):
        a += 1
    if keyboard.is_pressed('k'):
        a -= 1
    obs, rew, terminated, truncated,info, done = env.step(np.array([a]))
    if terminated:
        env.reset()
        pass
    
    time.sleep(0.5)
     
env.close()



