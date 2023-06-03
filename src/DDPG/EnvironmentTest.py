import IntersectionEnvironment
from IntersectionEnvironmentSumo import IntersectionEnv
from RingEnvironmentSimple import *
#from RingEnvironmentSimple import *
import numpy as np
import time
import matplotlib.pyplot as plt
from Vehicle import *

import os

envName = 'Ring'

myParams = IDM_CONTROLLER_DEFAULT_PARAMS.copy()
myParams['imperfection'] = 1/3000
myParams['a'] = 2
myParams['b'] = 2

#env = IntersectionEnv(
#    os.getcwd() + '/SumoIntersectionFiles',
#    0.05,
#    0,
#    timeStep=1/20,
#    maxIter=40000,
#    renderMode='human',
#)

NUM_ITER = 5000
TIME_STEP = 1/20

env = RingEnvAILess(
    50,
    25,
    1,
    20,
    2,
    TIME_STEP,
    NUM_ITER,
    IDMControllerParams=myParams,
    renderMode='human',
    saveStates=True
)


env.InitRender()

totalReward = 0
obs, info = env.reset()
done = False

i = 0
while not done:
    newObs, reward, terminated, truncated, info = env.step(np.zeros((0,)))
    done = terminated or truncated
    
    obs = newObs
    totalReward += reward
    
    #view = obs[0,:,0].reshape(-1)
    env.render()
    time.sleep(1/60)
    i+=1
    print(f'step:{i}', end = '\r')

print('episode done')

# fig, ax = plt.subplots(1,2)

# T = np.arange(0, TIME_STEP*NUM_ITER, TIME_STEP)
# for v in env.vehicles:
#     g = [p[0] % env.trackLenght for p in v.history]
#     for i in range(len(g)-1):
#         if g[i] > g[i+1]: g[i] = np.nan
#     ax[0].plot(T, g[:NUM_ITER])

# #plt.ylim((0,17.5))

# for v in env.vehicles:
#     g = [p[1] for p in v.history]
#     ax[1].plot(T, g[:NUM_ITER], alpha=0.7)

# ax[0].set_xlabel('Temps (s)')
# ax[1].set_xlabel('Temps (s)')
# ax[0].set_ylabel('Position (m)')
# ax[1].set_ylabel('Vistesse (m/s)')
# plt.show()

#print(sum([v.cumilativeAccel for v in env.vehicles]))
    

        

