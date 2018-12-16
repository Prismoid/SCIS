import scipy.special as spys
import scipy.misc as spym
import scipy.stats as spysts

import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# self-functions for calculating PDF
def calcUnitFindBlockProb(lamb):
    return 1 - np.exp(- lamb)
def calcWasteTime(PFork, lamb):
    return np.log(1 - PFork) / np.log(1 - calcUnitFindBlockProb(lamb))
def calcA(PFork, lamb):
    return lamb * calcWasteTime(PFork, lamb)
def calcC(PFork, lamb):
    return np.exp( calcA(PFork, lamb) * ( 1 - np.log(calcWasteTime(PFork, lamb)) ) )

# extended spys incomplete gamma function
def IncompGamma(a, x):
    return spys.gammaincc(a, x) * spys.gamma(a)
def ExtendedIncompGamma(a, x):
    return 1 / a * (spys.gamma(a + 1) * spys.gammaincc(a + 1, x) - np.power(x, a) * np.exp(- x))

# PDF(no delay and lower bound)
def expDist(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

def funcPDF(PFork, lamb, t):
    c = calcC(PFork, lamb)
    a = calcA(PFork, lamb)
    WT = calcWasteTime(PFork, lamb)
    return c * np.power(np.e, - lamb * (t - WT * np.log(t))) * lamb * (1 - WT * 1 / t)

# Calculating L2 distance
def calcPEDist(PFork, lamb):
    # 'cause often shows up
    Tw = calcWasteTime(PFork, lamb)
    # calculating each term (0-3)
    zero = 1 
    # first = np.exp(- lamb * Tw)
    # constant components
    A = 2 * lamb * calcC(PFork, lamb)
    a = lamb
    b = lamb * Tw
    c = Tw
    second = A * np.power(a, - (b + 1)) * ((a * c) * IncompGamma(b, a * Tw) -
                                           IncompGamma(b + 1, a * Tw))
    '''
    print("A: " + str(A))
    print("a: " + str(a))
    print("b: " + str(b))
    print("c: " + str(c))
    '''
    # constant components
    A = lamb * np.power(calcC(PFork, lamb), 2)
    a = lamb
    b = 2 * lamb * Tw
    c = Tw
    third = A * np.power(a, - (b + 1)) * (a*a*c*c *
                                          ExtendedIncompGamma(b - 1, a*Tw) -
                                          2*a*c * IncompGamma(b, a*Tw) +
                                          IncompGamma(b + 1, a*Tw))
    print("zero: " + str(zero))
    # print("first: " + str(first))
    print("second: " + str(second))
    print("third: " + str(third))
    print("A: " + str(A))
    print("a: " + str(a))
    print("b: " + str(b))
    print("c: " + str(c))
    print("IncompGamma(b-1, a*Tw): " + str(ExtendedIncompGamma(b-1, a*Tw)))
    return zero + second + third


# parameters
PFork = [0.001] # 0.25, 0.4]# 0.01, 0.025, 0.05, 0.1, 0.2, 0.4] # 0.062192 # 0.0178 # 0.0653
lamb = 1 / 15 # 0.0860662999989

# discretization
maxT = 5000
diff = 0.0001

print("TEST Start")

L2DistList = []
print("\hline")
print("フォーク確率 & PE 距離\\\ \hline \hline")
for i in range(len(PFork)):
    # L2 Distance
    L2DistList.append(calcPEDist(PFork[i], lamb))
    print(str(PFork[i]) + " & " + str(L2DistList[i]) + "\\\\")
print("\hline")

print(str(PFork))

plt.plot(PFork, L2DistList)
# plt.show()
