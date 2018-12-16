import numpy as np
import matplotlib.pyplot as plt
import math 
from numba import jit
from multiprocessing import Pool
from multiprocessing import Process

# 確率 0.6567
# start = 1500, end = 3001
# 5000回のデータ[0.0002, 0.0024, 0.0178, 0.061, 0.1612, 0.2722, 0.4056, 0.5174, 0.5844, 0.661, 0.7346, 0.7854, 0.8202, 0.8722, 0.8908, 0.9154]
# start = 1500, end = 3401
# 10000回ののデータ[0.0001, 0.0025, 0.0172, 0.0671, 0.16, 0.2744, 0.4034, 0.5068, 0.591, 0.668, 0.7321, 0.7804, 0.8302, 0.863, 0.8924, 0.9117, 0.9318, 0.9463, 0.9619, 0.9688]

# 1. シミュレーション環境を作成
@jit
def exePoW(blockLam, userLam, m, l, delta, delay):
    uBL = blockLam * delta # 処理単位あたりの解発見確率
    uUL = userLam  * delta # e.gh. 1 / 3600 * 1 / 100 = 360000
    # 処理限界回数
    nTry = 10800000
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
        rnd = np.random.rand() # 乱数
        rndLocal = np.random.rand() # 乱数
        rndAtker = np.random.rand() # 乱数
        if (rnd < uBL):
            # if (blockHeight % 100 == 0):
                # print("Block Mining Time: " + str(i / 100))
            blockHeight  += 1
            blockUpdTimes[0][blockHeight] = i
            blockUpdTimes[1][blockHeight] = math.ceil(delay * roundTime)
            '''
            while (True):
                normalRnd = np.random.normal(delay,delay/2)
                if (0 <= normalRnd and normalRnd <= delay): 
                    blockUpdTimes[1][blockHeight] = math.ceil(normalRnd * roundTime)
                    break
            '''
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
            while (True):
                normalRnd = np.random.normal(delay,delay/2)
                if (0 <= normalRnd and normalRnd <= delay):
                    PoWSendDelay = math.ceil(normalRnd * roundTime)
                    # print("PoWSendDelay: " + str(PoWSendDelay))
                    break
        if (blockUpdTimes[0][localBlockHeight + 1] + blockUpdTimes[1][localBlockHeight + 1] < i and blockUpdTimes[1][localBlockHeight + 1] != 0):
            localBlockHeight += 1
            # if (localBlockHeight % 10 == 0): 
                # print("localBlockHeight: " + str(localBlockHeight))
        if (i == PoWTime + PoWSendDelay and PoWFinFlag == 1):
            if (blockHeight - 1 - m <= PoWBlockHeight):
                return 0
            else:
                # print("HEHE")
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

# ラウンド
roundTime = 100
delta = 1 / roundTime # 1 / n 秒単位でPoW処理
start = 1500
end = 3401
stride = 100
delay = np.arange(start, end, stride) # [sec]
# np.1500 # 

# ブロックデータ
m = 256
l = 12

# 解発見確率(単位時間あたりの)
blockLam = 1 / 14.31
userLam  = 1 / 3600 # とりあえず1時間
        
# シミュレーション回数
cntNum = 10

# シミュレーションを実行

p = Pool(12)
result = p.map(calcProb, delay)
print(result)

# result = [0.0001, 0.0025, 0.0172, 0.0671, 0.16, 0.2744, 0.4034, 0.5068, 0.591, 0.668, 0.7321, 0.7804, 0.8302, 0.863, 0.8924, 0.9117, 0.9318, 0.9463, 0.9619, 0.9688]
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
