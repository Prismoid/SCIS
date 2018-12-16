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

def calcA(PFork, lamb):
    return lamb * calcWasteTime(PFork, lamb)

def calcC(PFork, lamb):
    return np.exp( calcA(PFork, lamb) * ( 1 - np.log(calcWasteTime(PFork, lamb)) ) )

def calcExpectTime(PFork, lamb):
    c = calcC(PFork, lamb)
    a = calcA(PFork, lamb)
    return c * np.power(lamb, -1 - a) * ( - a * spys.gammaincc(a + 1, a) * spys.gamma(a + 1) + spys.gammaincc(a + 2, a) * spys.gamma(a + 2))

# パラメータの設定
PFork = 0.062192 # 0.0178 # 0.0653
diff = - 0.0000001
lamb = np.arange(1/8, 1/15, diff)
expectedTime = 14.31 # 14.31

print(calcWasteTime(PFork, 1/14.31))
# データの作成
for i in range(len(lamb)):
    if (calcExpectTime(PFork, lamb[i]) >= expectedTime):
        print("lamb: " + str(lamb[i]))
        print("time(sync): " + str(1/lamb[i]))
        print("time(async): " + str(calcExpectTime(PFork, lamb[i])))
        break
#    
#     break
# print("beta: " + str(calcBeta(a, PFork, lamb[ans])))
