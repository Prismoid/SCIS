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
PFork = [0.01, 0.1, 0.4] # 0.062192 # 0.0178 # 0.0653
lamb = [1/15, 1 / 600] # 0.0860662999989
diff = 0.001

# データの作成
maxT = [1 / lamb[0] * 5, 1 / lamb[1] * 5]

# for making figs
meanT = ["T15", "T600"]
PFstr = ["PF001", "PF01", "PF04"]

for i in range(len(lamb)):
    for j in range(len(PFork)):
        # Case I: Lower bound of Propagation Delay CDF
        t1 = np.arange(0, calcWasteTime(PFork[j], lamb[i]), diff)
        t2 = np.arange(calcWasteTime(PFork[j], lamb[i]), maxT[i], diff)
        y1 = [0.0000001] * len(t1)
        y2 = funcPDF(PFork[j], lamb[i], t2) # for PDF
        t = np.concatenate((t1, t2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        plt.plot(t, y, label="Lower Bound of Propagation Delay CDF")
        # Case II: No Propagation Delay
        tExp = np.arange(0, maxT[i], diff)
        yExp = expDist(tExp, lamb[i])# =1/14.31)
        plt.plot(tExp, yExp, color="#ff0000", label="No Propagation Delay")
        # Figure settings
        plt.xlabel("Block Height Update Time [sec]", fontsize=18)
        plt.ylabel("PDF", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=14)
        # plt.rcParams["font.size"] = 18
        plt.xlim(xmin=0)
        plt.xlim(xmax=maxT[i] + 1)
        # plt.ylim(ymin=0.0)
        # plt.ylim(ymax=1.05)
        # Figure saving
        filename = "CompPDF_" + meanT[i] + PFstr[j]
        plt.ylim(bottom=0.0)
        # plt.savefig("images/" + filename, figsize=(30, 10), dpi=120,bbox_inches="tight")
        # plt.clf()
    plt.show()

# PDF
