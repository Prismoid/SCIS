import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import numpy.random as nprnd
import math 
import matplotlib.pyplot as plt
import time as tm

def func1(m, lamb, time):
    coeffi = np.power(math.exp(1), (- lamb * time))
    ret_sum = 0;
    for i in range(m - 1):
        tmp_log =  i * np.log(lamb * time) - math.lgamma(i + 1)
        ret_sum += np.power(math.exp(1), tmp_log)
    return  1 - coeffi * ret_sum

adjustCountUnder10 = 0
adjustCountOver20 = 0
# Delay CDF, Lower Bound
def rndFindBlockTime(lamb):
    time = nprnd.exponential(1/lamb)
    mult = 1 + 1 / 2048 * max(1 - time // 10, -99) # difficult
    return [time, lamb / mult]


# パラメータの設定
firstLamb = 1/14.31 # calculate time: 1/lamb
num = 3000000
time = [None] * (num + 1)
time[0] = 0
lambs = [None] * (num + 1)
lambs[0] = firstLamb
for i in range(num):
    tmpTime, lambs[i + 1] = rndFindBlockTime(lambs[i])
    time[i + 1] = tmpTime + time[i]

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


x = time
validNum = 250
xN = [0] * (num // validNum)
for i in range((num // validNum)):
    xN[i] = x[(i + 1) * validNum] - x[i * validNum]

plt.hist(xN, bins=1000,normed=True, histtype='step', cumulative=True, label='Ethereum Classic \nBC Data', color="green")

### Upper Bound                                                                                                                                                                      
# 時間データの作成
stride = 0.4
start = stride
end = 5500 # 100: 2800, 250: 5500
time2 = np.arange(start, end, stride)

# pre-orderが有効になるまでのブロック数
lamb1 = meanLamb
y1 = func1(validNum, lamb1, time2)

# グラフの描画                                                                                                                                                                       
plt.plot(time2, y1, label="Upper Bound", color="red")

plt.legend()

plt.show()
