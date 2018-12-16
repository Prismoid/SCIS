import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import numpy.random as nprnd
import math 
import matplotlib.pyplot as plt
import time as tm



adjustCountUnder10 = 0
adjustCountOver20 = 0
# Delay CDF, Lower Bound
def rndFindBlockTime(lamb):
    global adjustCountUnder10
    global adjustCountOver20
    time = nprnd.exponential(1/lamb)
    mult = 1 + 1 / 2048 * max(1 - time // 10, -99) # difficult
    if (np.absolute(1 / 2048 * max(1 - time // 10, -99)) > 0.01): 
        print(mult)
        tm.sleep(0.1)
    # mult = 1 + 1 / 2048 * max(2 - time // 9, -99) # difficulty
    if (time < 10):
        adjustCountUnder10 += 1
        # print("up"
    elif (time >= 20):
        adjustCountOver20 += 1
    # mult > 1: consensus時間を長くする, mult < 1: consensus時間を短くする
    return lamb / mult


# パラメータの設定
firstLamb = 1/14.31 # calculate time: 1/lamb
num = 500000
time = np.arange(0, num, 1)
lambs = [None] * num
lambs[0] = firstLamb
for i in range(num - 1):
    lambs[i + 1] = rndFindBlockTime(lambs[i])

print("最大: " + str(max(lambs)) + ", 時間: ( " + str(1/max(lambs)) + " )")
print("最小: " + str(min(lambs)) + ", 時間: ( " + str(1/min(lambs)) + " )")
print("平均: " + str(np.mean(lambs, axis=0)))
print("標準偏差: " + str(np.std(lambs, axis=0)))
meanLamb = np.mean(lambs, axis=0)
highLamb = (np.mean(lambs, axis=0) + 3 * np.std(lambs, axis=0)) / np.mean(lambs, axis=0) # 難易度低い
lowLamb  = (np.mean(lambs, axis=0) - 3 * np.std(lambs, axis=0)) / np.mean(lambs, axis=0) # 難易度高い
meanTime  = 1/meanLamb
highTime = 1/(np.mean(lambs, axis=0) - 3 * np.std(lambs, axis=0)) # 難易度高い
lowTime  = 1/(np.mean(lambs, axis=0) + 3 * np.std(lambs, axis=0)) # 難易度低い
print("平均時間: " + str(meanTime))
print("高難易度の時間: " + str(highTime))
print("低難易度の時間: " + str(lowTime))

print("\n難易度調整(10sec以下): " + str(adjustCountUnder10))
print("難易度調整(20sec以上): " + str(adjustCountOver20))

plt.xlabel("Elapsed Blocks", fontsize=20)
plt.ylabel("lamb", fontsize=20)

plt.tick_params(labelsize = 20)
plt.plot(time, lambs, label="TEST")
plt.legend()

plt.show()
