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
PFork = 0.4 # 0.061564 # 4600000-5100000 data
expectedTime = 16.32# 14.296378 # 4600000-5100000 data
diff = 0.000001
lamb = 1/expectedTime * (1 + PFork)

print("遅延により無駄にされた計算資源の時間: " + str(calcWasteTime(PFork, lamb)))
print("最悪な伝播をした場合の期待時間: " + str(calcExpectTime(PFork, lamb)))
print("下限の伝播をした場合、何倍遅いか: " + str(calcExpectTime(PFork, lamb) / expectedTime))


