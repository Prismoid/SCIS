import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# Mining CDF
def calcUnitFindBlockProb(lamb):
    return 1 - np.exp(- lamb)

def calcWasteTime(PFork, lamb):
    return np.log(1 - PFork) / np.log(1 - calcUnitFindBlockProb(lamb))

def calcA(PFork, lamb):
    return lamb * calcWasteTime(PFork, lamb)

def calcC(PFork, lamb):
    return np.exp( calcA(PFork, lamb) * ( 1 - np.log(calcWasteTime(PFork, lamb)) ) )

def calcMiningCDFLB(PFork, lamb, wasteTime, coeffi, lambXlamb, time):
    return 1.0 - coeffi * np.exp(- lamb * (time - wasteTime * np.log(time)))



# パラメータの設定
PFork = 0.062192 # 0.0178 # 0.0653
lamb = 1/14.31
wasteTime = calcWasteTime(PFork, lamb)
coeffi = calcC(PFork, lamb)
lambXlamb = calcA(PFork, lamb)

diff = 0.01
start = wasteTime # calcWasteTime(PFork, lamb)
end = 60
time = np.arange(start, end, diff)
y = calcMiningCDFLB(PFork, lamb, wasteTime, coeffi, lambXlamb, time)

expectedTime = 14.31 # 14.31


plt.xlabel("Time")
plt.ylabel("CDF")

plt.xlim(xmin=0)
plt.xlim(xmax=end + 1) # 10100)                                                                                                                                                    
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)

plt.plot(time, y, label="CDF Lower Bound")
plt.legend()

plt.show()
