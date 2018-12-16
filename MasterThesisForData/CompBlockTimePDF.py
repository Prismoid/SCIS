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
    a = calcA(PFork, lamb)
    WT = calcWasteTime(PFork, lamb)
    return c * np.power(np.e, - lamb * (t - WT * np.log(t))) * lamb * (1 - WT * 1 / t)

def expDist(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

# パラメータの設定
PFork = 0.05 # 0.062192 # 0.0178 # 0.0653 
lamb = 1 / 600 # 0.0860662999989 
diff = 0.0001

# データの作成
maxT = 3000
t = np.arange(calcWasteTime(PFork, lamb), maxT, diff)
y = [None] * len(t)
y = funcPDF(PFork, lamb, t) # for PDF
'''
for i in range(len(t)): 
    y[i] = funcPDF(PFork, lamb, t) # for PDF
'''
yExp = expDist(t, lamb)# =1/14.31)

plt.plot(t, yExp, color="#ff0000", label="Synchronized Network")
        
# グラフの描画
# CDF
'''
plt.plot(t, y, label="Data")
plt.xlabel("Time")
plt.ylabel("CDF")

plt.legend()

plt.xlim(xmin=0)
plt.xlim(xmax=maxT + 1) 
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)
plt.show()
'''
# Kullback Leibler
probY = (y * diff)[1:-1]
probYExp = (yExp * diff)[1:-1]
print(probY)
print(spysts.entropy(probYExp, probY))


# PDF
plt.plot(t, y, label="Asynchronized Network")
plt.xlabel("Block Time [sec]")
plt.ylabel("PDF")

plt.legend()

# plt.xlim(xmin=0)
# plt.xlim(xmax=maxT + 1) 
# plt.ylim(ymin=0.0)
# plt.ylim(ymax=1.05)
plt.show()
