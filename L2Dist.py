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
def calcL2Dist(PFork, lamb):
    # 'cause often shows up
    Tw = calcWasteTime(PFork, lamb)
    # calculating each term (0-3)
    zero = lamb / 2 * (1 - np.exp(- 2 * lamb * Tw))
    first = lamb / 2 * np.exp(- 2 * lamb * Tw)
    # constant components
    A = 2 * lamb * lamb * calcC(PFork, lamb)
    a = 2 * lamb
    b = lamb * Tw
    c = Tw
    second = A * np.power(a, - (b + 1)) * ((a * c) * IncompGamma(b, a * Tw) -
                                           IncompGamma(b + 1, a * Tw))
    # constant components
    A = np.power(lamb * calcC(PFork, lamb), 2)
    a = 2 * lamb
    b = 2 * lamb * Tw
    c = Tw
    third = A * np.power(a, - (b + 1)) * (a*a*c*c *
                                          ExtendedIncompGamma(b - 1, a*Tw) -
                                          2*a*c * IncompGamma(b, a*Tw) +
                                          IncompGamma(b + 1, a*Tw))

    print(str(zero))
    print(str(first))
    print(str(second))
    print(str(third))
    '''
    print("A: " + str(A))
    print("a: " + str(a))
    print("b: " + str(b))
    print("c: " + str(c))
    print("IncompGamma(b-1, a*Tw): " + str(IncompGamma(b-1, a*Tw)))
    '''
    return zero + first + second + third

# parameters
PFork = [0.001] #, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4] # 0.062192 # 0.0178 # 0.0653
lamb = 1 / 600 # 0.0860662999989

# discretization
maxT = 5000
diff = 0.0001

print("TEST Start")

L2DistList = []
print("\hline")
print("フォーク確率 & $L^2$ 距離\\\ \hline \hline")
for i in range(len(PFork)):
    # L2 Distance
    L2DistList.append(calcL2Dist(PFork[i], lamb))
    print(str(PFork[i]) + " & " + str(L2DistList[i]) + "\\\\")
print("\hline")

print(str(PFork))

plt.plot(PFork, L2DistList)
plt.show()
