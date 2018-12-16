import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# PDF, RumorSpreading
def calcUnitFindBlockProb(lamb):
    return 1 - np.exp(- lamb)

def calcWasteTime(PFork, lamb):
    return np.log(1 - PFork) / np.log(1 - calcUnitFindBlockProb(lamb))

def calcCDFLB(PFork, lamb, time):
    Xr = calcWasteTime(PFork, lamb)
    return 1 - Xr / time



# パラメータの設定
PFork = 0.062192 # 0.0178 # 0.0653
lamb = 1/13.0
diff = 0.01
start = calcWasteTime(PFork, lamb)
end = 10
time = np.arange(start, end, diff)
y = calcCDFLB(PFork, lamb, time)

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
