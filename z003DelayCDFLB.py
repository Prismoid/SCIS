import scipy.special as spys
import scipy.misc as spym # コンビネーション用

import numpy as np
import math
import matplotlib.pyplot as plt

# Delay CDF, Lower Bound
def calcUnitFindBlockProb(lamb):
    return 1 - np.exp(- lamb)

def calcWasteTime(PFork, lamb):
    return np.log(1 - PFork) / np.log(1 - calcUnitFindBlockProb(lamb))

def calcCDFLB(PFork, lamb, time):
    Xr = calcWasteTime(PFork, lamb)
    return 1 - Xr / time

# e.g. CDF, RumorSpreading
def func(a, beta, t):
    c1 = a/(1-a)
    return c1 / (np.power(np.e, -beta * t) + c1)

# パラメータの設定
beta = 5.52839875415 # 1.92965593471 # 最終予定: 10.2261233386
a = 0.01 # 最終予定: 1/5000
diff = 0.0001

# データの作成
'''
maxT = 8
t = np.arange(0, maxT, diff)
y = func(a, beta, t) # for CDF
plt.plot(t, y, label="Example", color='red')
'''
# パラメータの設定
PFork = 0.05
lamb = 1/600
diff = 0.1
start = calcWasteTime(PFork, lamb)
end = 420
time = np.arange(start, end, diff)
y = calcCDFLB(PFork, lamb, time)


plt.xlabel("Time [sec]", fontsize=20)
plt.ylabel("CDF", fontsize=20)
plt.title("Block Propagation", fontsize=20)

plt.xlim(xmin=0)
plt.xlim(xmax=end + 1) # 10100)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)
plt.tick_params(labelsize = 20)
plt.plot(time, y) # label="CDF's Lower Bound")
# plt.legend()

plt.show()
