import scipy.special as spys
import numpy as np
import matplotlib.pyplot as plt

### parameters to characterize the propagation delay ###
def calcUnitFindBlockProb(lamb):
    return 1 - np.exp(- lamb)

def calcWasteTime(PFork, lamb):
    return np.log(1 - PFork) / np.log(1 - calcUnitFindBlockProb(lamb))

def calcA(PFork, lamb):
    return lamb * calcWasteTime(PFork, lamb)

def calcC(PFork, lamb):
    return np.exp( calcA(PFork, lamb) * ( 1 - np.log(calcWasteTime(PFork, lamb)) ) )

def calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, x):
    return - wasteTime * spys.lambertw(- np.power((1 - x) / coeffi, 1 / lambXlamb) /  wasteTime, -1)

# input values
lamb  = 1/14.31
PFork = 0.062192

# each parameters
diff = 0.001
# xVal = np.arange(0, 1, 0.001)
# xVal = np.random.rand(100000)
wasteTime = calcWasteTime(PFork, lamb)
coeffi = calcC(PFork, lamb)
lambXlamb = calcA(PFork, lamb)

print(wasteTime)
print(coeffi)
print(lambXlamb)

blockNum = 100
y = [0] * 100000
for i in range(blockNum):
    xVal = np.random.rand(100000)
    y += calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, xVal)

# print(xVal)
# print(y)

def calcMiningCDFLB(PFork, lamb, wasteTime, coeffi, lambXlamb, time):
    return 1.0 - coeffi * np.exp(- lamb * (time - wasteTime * np.log(time)))
# print(calcMiningCDFLB(PFork, lamb, wasteTime, coeffi, lambXlamb, y))


plt.xlabel("Histogram [s]")
plt.ylabel("PDF")
plt.title("Block vs. time")
plt.xlim([0, 4000])
plt.ylim([0, 1.08])
plt.hist(y, bins=10,normed=True, color="#000080", ec='black')
plt.hist(y, bins=1000, normed=True, cumulative=True, histtype='step', label='CDF') # 
plt.show()

print(min(y))
'''
plt.xlabel("x-axis")
plt.ylabel("y-axis")

plt.plot(xVal, y, label="LambertW Test")
plt.legend()

plt.show()

'''
