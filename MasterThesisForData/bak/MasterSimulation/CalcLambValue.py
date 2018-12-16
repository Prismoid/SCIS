import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# PDF, RumorSpreading
def calcBeta(a, PFork, lamb):
    return np.log(1/a) * np.log(1 - lamb) / np.log(1 - PFork)

def calcExpectTime(a, PFork, lamb):
    beta = calcBeta(a, PFork, lamb)
    z = - (1 - a) / a
    hyp2f1 = spys.hyp2f1(lamb/beta, lamb/beta, lamb/beta + 1, z)
    exp = np.power(1/a, lamb/beta)
    return 1 / lamb * exp * hyp2f1

'''
def myhyp2f1(alpha, z):
    retSum  = 1
    first   = 1
    index   = 0
    prevVal = first
    while (True):
        if (np.absolute((a + index) / (a + index + 1) * z / (index + 1)) < 1 / 1000):
            break
        # if (index % 1000 == 0):
            # print("HEHEHE")
            # print(np.absolute((a + index) / (a + index + 1) * z / (index + 1)))
        nextVal = ((a + index) / (a + index + 1) * z / (index + 1)) * prevVal
        retSum += nextVal
        if (index == 161):
            print(retSum)
        if (math.isnan(retSum)):
            sys.exit(3)
            # print(index)
            # print("BUGBUG")
        prevVal = nextVal
        index += 1
    return retSum
'''

# パラメータの設定
PFork = 0.062192 # 0.0178 # 0.0653
a = 0.01 # 1 / 5000 # 5000ノード
diff = 0.00001
lamb = np.arange(1/15, 1/8, diff)
print(lamb)
expectedTime = 14.31 # 14.31

# データの作成
flag = 0
ans  = 0
for i in range(len(lamb)):
    if (calcExpectTime(a, PFork, lamb[i]) <= expectedTime and flag == 0):
        print("lambda: " + str(lamb[i]))
        print("time(sync): " + str(1/lamb[i]))
        ans = i
        flag = 1
        break
    
print("beta: " + str(calcBeta(a, PFork, lamb[ans])))
