import matplotlib.pyplot as plt
import numpy as np

path = "C:/Users/Tribik/Documents/Mohamed/Etudes/MP/TIPE/Code/Ring/agents/agent Pendalum(2023-4-4 14.36.13)/log.txt"

f = open(path, 'r')

L = []

for ligne in f:
    rewardText = ligne.split('|')[1]
    reward = float(rewardText.split(':')[1])
    L.append(reward)

L = np.array(L)
A = np.array([ sum(L[max(0,i - 100): i+1]) / min(100,i+1) for i in range(len(L)) ])

V = np.array([ sum((L[max(0,i - 100): i+1] - A[i])**2) / min(100,i+1) for i in range(len(L)) ])

sigma = np.sqrt(V)

plt.plot(A+sigma,alpha=0.5)
plt.plot(A-sigma,alpha=0.5)
plt.plot(A)
plt.show()