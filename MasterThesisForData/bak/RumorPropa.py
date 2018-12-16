import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt
import sys

# PDF, RumorSpreading
# beta: 9 -> 1秒間に期待値9ブロック(1ブロックあたり)で伝播(N - I)/N ~ 1の場合に
def stochasticProcess(beta, maxNode, nowNode, diff):
    sucProb = beta * (maxNode - nowNode) / maxNode * diff
    retSum  = 0
    for i in range(nowNode):
        if (sucProb > np.random.rand()):
            print("Excute")
            retSum += 1
    return retSum

# パラメータの設定
beta = 9 # 0.76282514433 # 9.76219788788
diff = 0.01 # 0.1 seconds
time = 0.1 # 100 msec(1 block propagation)

# データの作成
maxT = 20 # 50 seconds
maxNode = 5000
startNode = 1
t = np.arange(0, maxT, diff) # for Time
y = [0] * len(t) # for histogram
add = [0] * (len(t) + int(time/diff))

y[0] = startNode
for j in range(len(t) - 1):
    add[j + int(time / diff)] = stochasticProcess(beta, maxNode, y[j], diff)
    if (j > 0): 
        y[j + 1] = y[j] + add[j - 1]

print(y)

for j in range(len(t) - 1):
    if (y[j + 1] - y[j] > 2):
        print("Dekai")

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
