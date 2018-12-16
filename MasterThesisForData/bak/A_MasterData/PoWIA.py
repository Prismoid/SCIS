import numpy as np
import matplotlib.pyplot as plt

# 1. シミュレーション環境を作成

def findNonce(blockLam, userLam, atkLam, delta, delay):
    uBL = blockLam * delta # 処理単位あたりの解発見確率
    uUL = userLam  * delta
    uAL = atkLam   * delta
    # 処理限界回数
    nTry = 1080000
    # global data
    blockHeight   = 0
    blockUpdTimes = np.array([0] * nTry)
    # local data
    localBlockHeight = 0
    PoWBlockHeight   = 0 - 1 - 12
    PoWTime          = 0
    PoWFinFlag = 0
    PoWAtkFlag = 0
    for i in range(nTry):
        rnd = np.random.rand() # 乱数
        rndLocal = np.random.rand() # 乱数
        rndAtker = np.random.rand() # 乱数
        if (rnd < uBL):
            if (blockHeight % 100 == 0):
                # print("Block Mining Time: " + str(i / 100))
            blockHeight  += 1
            blockUpdTimes[blockHeight] = i
        if (rndLocal < uUL and PoWFinFlag == 0):
            # print("PoW Success Time: " + str(float(i) / 100))
            PoWBlockHeight = localBlockHeight - 1 - 12
            if (PoWBlockHeight < 0):
                PoWFinFlag = 0
                continue
            #  print("PoWBlockHeight: " + str(PoWBlockHeight))
            PoWTime    = i
            PoWFinFlag = 1
        if (rndAtker < uAL and PoWFinFlag == 1):
            PoWAtkFlag = 1

        if (blockUpdTimes[localBlockHeight + 1] + (20 * 100) < i and blockUpdTimes[localBlockHeight + 1] != 0):
            localBlockHeight += 1
            if (localBlockHeight % 1000 == 0):
                print("localBlockHeight: " + str(localBlockHeight))
        if (i == PoWTime + (20 * 100) and PoWFinFlag == 1):
            if (blockHeight - 1 - 256 <= PoWBlockHeight and PoWAtkFlag == 0):
                return 0
            elif (blockHeight - 1 - 256 <= PoWBlockHeight and PoWAtkFlag == 1): 
                return 1/2
            else:
                return 1
    return 1
    
# ラウンド
roundTime = 100
delta = 1 / roundTime # 1 / n 秒単位でPoW処理
delay = 20 # [sec]

# 解発見確率(単位時間あたりの)
blockLam = 1 / 15
userLam  = 1 / 3600 / 5
atkLam   = userLam * 100 # 100倍の資源
        
# 乱数
cntNum = 1

# シミュレーションを実行
result = np.array([None]*cntNum)
print(result)
for i in range(cntNum):
    result[i] = findNonce(blockLam, userLam, atkLam, delta, delay) 
print(result)
plt.xlabel("Time since last block [s]")
plt.ylabel("PDF")
plt.title("Block vs. time")
#plt.xlim([0, 120])
#plt.ylim([0, 1.0])

# plt.plot(, Out, color="#ff0000")
# plt.plot
plt.hist(result, bins=50, normed=True, color="#000080", ec='black') # bins=50
plt.show()

        


