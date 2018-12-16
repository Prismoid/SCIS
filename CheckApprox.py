import scipy.special as spys
import scipy.misc as spym
import scipy.stats as spysts

import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# self-functions for calculating PDF(theoretical and numerical)
def ExtendedIncompGamma(a, x):
    return 1 / a * (spys.gamma(a + 1) * spys.gammaincc(a + 1, x) - np.power(x, a) * np.exp(- x))
def ReimannSumIncompGamma(a, x, maxT=1000):
    diff = 0.000001
    t = np.arange(x, maxT, diff)
    y = np.power(t, a - 1) * np.exp(-t)
    sum = np.sum(y, axis=0)
    return sum * diff

# parameters
PFork = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4] # 0.062192 # 0.0178 # 0.0653
lamb = 1 / 600 # 0.0860662999989

# discretization
maxT = 5000
diff = 0.0001

print("TEST Start")

print(ExtendedIncompGamma(-0.90, 0.00001))
print(ReimanSumIncompGamma(-0.90, 0.00001))
