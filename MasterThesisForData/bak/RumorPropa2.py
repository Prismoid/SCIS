import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt
import sys

# PDF, RumorSpreading
# beta: 9 -> 1秒間に期待値9ブロック(1ブロックあたり)で伝播(N - I)/N ~ 1の場合に
def stochasticProcess(beta, maxNode, nowNode, informNode, diff):
    sucProb = beta * nowNode * (maxNode - informNode) / maxNode * diff
    retSum  = 0
    if (sucProb > np.random.rand()):
        retSum = 1
    return retSum

# パラメータの設定
beta = 0.5 # 0.76282514433 # 9.76219788788
intVal = 10000
diff = 1 / intVal # 0.1 seconds, poisson process, 1回しか起きない
blockTime = 0.01 # 0.1 seconds

# データの作成
maxT = 50 # 50 seconds
maxNode = 200
startNode = 1
t = np.arange(0, maxT, diff)
y = [0] * int(maxT * intVal)  # for histogram
delay = [0] * int(maxT * intVal + 2 * blockTime * intVal)  # for histogram
print(delay)
y[0] = startNode
informNode = 1
for j in range(int(maxT * intVal) - 1):
    if (j % intVal == 0): 
        print(str(int(j / intVal)) + " [sec]. " + str(y[j]))
    delay[j + int(blockTime * intVal)] = stochasticProcess(beta, maxNode, y[j], informNode, diff)
    if (delay[j + int(blockTime * intVal)] == 1):
        informNode += 1
    if (informNode > 5000):
        delay[j + int(blockTime * intVal)] = 0
    y[j + 1] = y[j] + delay[j]

'''
for j in range(len(t) - 1):
    if (y[j + 1] - y[j] > 2):
        print("Dekai")
'''

# グラフの描画
# CDF

plt.plot(t, y, label="Data")
plt.xlabel("Time")
plt.ylabel("CDF")

plt.legend()

plt.xlim(xmin=0)
plt.xlim(xmax=maxT + 1) 
plt.ylim(ymin=0.0)
plt.ylim(ymax=max(y) + 0.01)
plt.show()
