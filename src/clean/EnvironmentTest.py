from RingEnvironment import *
import numpy as np
import matplotlib.pyplot as plt

myParams = IDM_CONTROLLER_DEFAULT_PARAMS.copy()
myParams['imperfection'] = 1/2000
myParams['a'] = 2
myParams['b'] = 2

NUM_ITER = 5000
TIME_STEP = 1/20

env = RingEnv(
    radius=50,
    numVehicles=25,
    numVehiclesAuto=2,
    maxSpeed=20,
    maxAccel=2,
    timeStep=TIME_STEP,
    maxIter=NUM_ITER,
    IDMControllerParams=myParams,
    renderMode='human',
    saveStates=True
)

env.reset()
done = False

i = 0
while not done:
    terminated = env.step()
    done = terminated 

    i+=1
    print(f'step:{i}', end = '\r')

print('episode done')

fig, ax = plt.subplots(1,2)

T = np.arange(0, TIME_STEP*NUM_ITER, TIME_STEP)
for v in env.vehicles:
    g = [p[0] % env.trackLenght for p in v.history]
    for i in range(len(g)-1):
        if g[i] > g[i+1]: g[i] = np.nan   # Pour eviter de tracer les discontinuite
    ax[0].plot(T, g[:NUM_ITER])

for v in env.vehicles:
    g = [p[1] for p in v.history]
    ax[1].plot(T, g[:NUM_ITER], alpha=0.7)

ax[0].set_xlabel('Temps (s)')
ax[1].set_xlabel('Temps (s)')
ax[0].set_ylabel('Position (m)')
ax[1].set_ylabel('Vistesse (m/s)')

plt.show()    

        

