import scipy.special as spys
import scipy.misc as spym # コンビネーション用
import scipy.stats as spysts

import numpy as np
import math
import matplotlib.pyplot as plt

# PDF, RumorSpreading
def calcUnitFindBlockProb(lamb):
    return 1 - np.exp(- lamb)

def calcWasteTime(PFork, lamb):
    return np.log(1 - PFork) / np.log(1 - calcUnitFindBlockProb(lamb))

def calcA(PFork, lamb):
    return lamb * calcWasteTime(PFork, lamb)

def calcC(PFork, lamb):
    return np.exp( calcA(PFork, lamb) * ( 1 - np.log(calcWasteTime(PFork, lamb)) ) )

def funcPDF(PFork, lamb, t):
    c = calcC(PFork, lamb)
    WT = calcWasteTime(PFork, lamb)
    return c * np.power(np.e, - lamb * (t - WT * np.log(t))) * lamb * (1 - WT * 1 / t)

def funcCDF(PFork, lamb, t):
    c = calcC(PFork, lamb)
    WT = calcWasteTime(PFork, lamb)
    return 1 - c * np.power(np.e, - lamb * (t - WT * np.log(t)))


def expDist(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

# パラメータの設定
lamb = 1 / 600 # 0.0860662999989
diff = 0.01

# 時間のプロット
maxT = 2200
t1 = np.arange(0, maxT, diff)

### グラフの描画
## Synchronized
yExp = 1 - np.exp(- lamb * t1)
plt.plot(t1, yExp, color="#ff0000", label="Synchronized, $P_F = 0.00$")
## Asynchronized
# First
PFork = 0.1 # 0.062192 # 0.0178 # 0.0653
t2 = np.arange(calcWasteTime(PFork, lamb), maxT, diff)
y = funcCDF(PFork, lamb, t2) # for PDF
plt.plot(t2, y, label="Asynchronized, $P_F = 0.10$ (Lower Bound)")
# Second
PFork = 0.05 # 0.062192 # 0.0178 # 0.0653
t2 = np.arange(calcWasteTime(PFork, lamb), maxT, diff)
y = funcCDF(PFork, lamb, t2) # for PDF
plt.plot(t2, y, label="Asynchronized, $P_F = 0.05$ (Lower Bound)")

plt.xlabel("Time for the longest chain to grow by 1 block", fontsize=18)
plt.ylabel("CDF", fontsize=18)

plt.tick_params(labelsize=18)
plt.legend(fontsize=18)
plt.rcParams["font.size"] = 18

plt.xlim(xmin=0)
plt.xlim(xmax=maxT + 1)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)
plt.legend(loc='lower right', fontsize=18)
plt.show()
