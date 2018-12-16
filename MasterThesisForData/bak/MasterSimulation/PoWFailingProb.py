import numpy as np
import matplotlib.pyplot as plt
import math 
from numba import jit
from multiprocessing import Pool
from multiprocessing import Process
import time
import sys

# 確率 0.6567
# start = 1500, end = 3001
# 5000回のデータ[0.0002, 0.0024, 0.0178, 0.061, 0.1612, 0.2722, 0.4056, 0.5174, 0.5844, 0.661, 0.7346, 0.7854, 0.8202, 0.8722, 0.8908, 0.9154]
# start = 1500, end = 3401
# 10000回ののデータ[0.0001, 0.0025, 0.0172, 0.0671, 0.16, 0.2744, 0.4034, 0.5068, 0.591, 0.668, 0.7321, 0.7804, 0.8302, 0.863, 0.8924, 0.9117, 0.9318, 0.9463, 0.9619, 0.9688]

def funcCDF(beta, a, t):
    c1 = a/(1 - a)
    return c1 / (np.exp(-beta * t) + c1)  

timeForCDF = np.arange(0, 200, 1)
timeForCDF = timeForCDF / 100
resultCDF = funcCDF(10.2261233386, 1/5000, timeForCDF)

# 1. シミュレーション環境を作成
@jit
def exePoW(blockLam, userLam, m, l, delta, delay):
    uBL = blockLam * delta # 処理単位あたりの解発見確率
    uUL = userLam  * delta # e.gh. 1 / 3600 * 1 / 100 = 360000
    # 処理限界回数
    nTry = 10800000 # 1回あたり0.01秒で計算
    # global data
    blockHeight   = 0
    blockUpdTimes = np.zeros((2, nTry))
    # local data
    localBlockHeight = 0
    PoWBlockHeight   = 0 - 1 - 12
    PoWTime          = 0
    PoWFinFlag = 0
    PoWSendDelay = 0
    # 乱数
    normalRnd = 0
    for i in range(nTry):
        uBL = 0
        if (i < 200): 
            uBL = blockLam * resultCDF[i]  * delta
        else:
            uBL = blockLam * delta
        rnd = np.random.rand() # 乱数
        rndLocal = np.random.rand() # 乱数
        if (rnd < uBL):
            # if (blockHeight % 100 == 0):
            #    print("Block Mining Time: " + str(i / 100))
            blockHeight  += 1
            blockUpdTimes[0][blockHeight] = i
            blockUpdTimes[1][blockHeight] = math.ceil(delay * roundTime)
        if (rndLocal < uUL and PoWFinFlag == 0):
            # print("PoW Success Time: " + str(float(i) / 100))
            PoWBlockHeight = localBlockHeight - 1 - l
            if (PoWBlockHeight < 0):
                PoWFinFlag = 0
                continue
            # print("PoWBlockHeight: " + str(PoWBlockHeight))
            # print("blockHeight: " + str(blockHeight))
            PoWTime    = i
            PoWFinFlag = 1
            PoWSendDelay = math.ceil(delay * roundTime)
        if (blockUpdTimes[0][localBlockHeight + 1] + blockUpdTimes[1][localBlockHeight + 1] < i and blockUpdTimes[1][localBlockHeight + 1] != 0):
            localBlockHeight += 1
            # if (localBlockHeight % 10 == 0): 
                # print("localBlockHeight: " + str(localBlockHeight))
        if (i == PoWTime + PoWSendDelay and PoWFinFlag == 1):
            if (blockHeight - 1 - m <= PoWBlockHeight):
                return 0
            else:
                return 1
    return 3

@jit
def calcProb(delay):
    result = [None] * cntNum
    for i in range(cntNum):
        result[i] = exePoW(blockLam, userLam, m, l, delta, delay)
        if (i % 1000 == 0):
            print(str(i) + " loop")
    return sum(result) / cntNum

# 時間測定
start = time.time()

# ラウンド
roundTime = 100
delta = 1 / roundTime # 1 / n 秒単位でPoW処理
start = 1000
end = 1300
stride = 50
delay = np.arange(start, end, stride) # [sec]
# np.1500 # 

# ブロックデータ
m = 128
l = 12

# 解発見確率(単位時間あたりの)
blockLam = 1 / 13.477 # ネットワークの計算資源は1/13.477で計算
userLam  = 1 / 3600 # とりあえず1時間
        
# シミュレーション回数
cntNum = 10000

# シミュレーションを実行

p = Pool(12)
result = p.map(calcProb, delay)
print(result)
elapsed_time = time.time() - start
print(str(elapsed_time) + "[sec]")

# m = 256
# start = 1100, end = 2000, stride = 50
# [0.0, 0.0, 0.0, 0.0, 0.0002, 0.0017, 0.0078, 0.0259, 0.0729, 0.1708, 0.3278, 0.5083, 0.6875, 0.8275, 0.9191, 0.9706, 0.9895, 0.9968]
# m = 192
# start = 750, end = 1600, stride = 50
# [0.0, 0.0, 0.0, 0.0, 0.0011, 0.0051, 0.0245, 0.0916, 0.2197, 0.4173, 0.6382, 0.8154, 0.9218, 0.9729, 0.9924, 0.9971, 0.9995]
# m = 128
# start = 400, end = 1300, stride = 50
# [0.0, 0.0, 0.0, 0.0001, 0.0027, 0.0229, 0.1073, 0.315, 0.5783, 0.805, 0.9346, 0.9826, 0.9972, 0.9995, 1.0, 1.0, 1.0, 1.0]

# [0.0, 0.0, 0.0003, 0.0069, 0.0752, 0.3212, 0.6815, 0.922, 0.9894, 0.9991, 0.9999]
plt.scatter(delay, result, c='r')
plt.show()
'''
plt.xlabel("Time since last block [s]")
plt.ylabel("PDF")
plt.title("Block vs. time")
#plt.xlim([0, 120])
#plt.ylim([0, 1.0])

# plt.plot(, Out, color="#ff0000")
# plt.plot
plt.hist(result, bins=50, normed=True, color="#000080", ec='black') # bins=50
plt.show()
'''

