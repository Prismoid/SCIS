import scipy.special as spys
import scipy.misc as spym # コンビネーション用
import scipy.stats as spysts

import sys
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
    a = calcA(PFork, lamb)
    WT = calcWasteTime(PFork, lamb)
    return c * np.power(np.e, - lamb * (t - WT * np.log(t))) * lamb * (1 - WT * 1 / t)

def expDist(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

# パラメータの設定
PFork = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4] # 0.062192 # 0.0178 # 0.0653
lamb = 1 / 600 # 0.0860662999989

# discretization
maxT = 5000
diff = 0.0001

print("TEST Start")
# データの作成

KLD = []
for i in range(len(PFork)):
    # x-axis(time)
    t = np.arange(calcWasteTime(PFork[i], lamb), maxT, diff)
    # Synchronized
    yExp = expDist(t, lamb)
    probYExp = (yExp * diff)[1:-1]
    # Asynchronoized
    y = funcPDF(PFork[i], lamb, t)
    probY = (y * diff)[1:-1]
    # Kullback Leibler
    KLD.append(spysts.entropy(probYExp, probY))
    print(str(PFork[i]) + " & " + str(KLD[i]))

print(str(PFork))
print(str(KLD))
