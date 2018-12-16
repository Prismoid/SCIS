import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# m - l - nブロック以上のブロックが生成される確率
def func(m, n, l, lamb, Delay): 
    forCoeffi = - lamb * Delay
    ret_sum = 0;
    for i in range(m - n - l + 1):
        tmp_log =  i * np.log(lamb * Delay) - spys.loggamma(i + 1)
        ret_sum += np.power(np.e, forCoeffi + tmp_log)
    return  ret_sum

#### パラメータの設定 ####
# ハッシュレート[block/sec]
meanLamb = 14.296378 # 4600000-5100000 data
PFork = 0.061564 # 4600000-5100000 data

lamb = 1 / meanLamb * (1 + PFork) # upper bound(12/21): 11.6189495774

# 有効ブロックハッシュ(Ethereumが辿れる前のブロックハッシュ数)
n = 100
m = [n * 3, n * 4, n * 5]
l = [None] * len(m)
for i in range(len(m)): 
    l[i] = np.arange(10, m[i] - n, 10)
time = [None] * len(m)
for i in range(len(m)):
    time[i] = [None] * len(l[i])

# データの作成
# 遅延時間
for i in range(len(m)): 
    for j in range(len(l[i])):
        now = 0
        timeForCalc = 1
        incr = 500
        flag = 0
        while (np.abs(now - 0.001) > 0.001 * 0.0001):
            now = func(m[i], n, l[i][j], lamb, timeForCalc)
            #        print(now)
            timeForCalc += incr
            if (flag == 0 and now < 0.001):
                flag = 1
                incr = - 1 * incr / 2
            elif (flag == 1 and now > 0.001):
                flag = 0
                incr = - 1 * incr / 2
            time[i][j] = timeForCalc

print(m)
print(l)
print(time)
maxTime = time[2][0] * 1.05

# グラフの描画


plt.plot(l[0], time[0], label="m = n * 3")
plt.plot(l[1], time[1], label="m = n * 4")
plt.plot(l[2], time[2], label="m = n * 5")

plt.xlabel("Waiting blocks until a registaration transaction is included in the BC", fontsize=20)
plt.ylabel("Granted delay time [sec]", fontsize=20)
plt.legend(loc='upper right', fontsize=20)
plt.tick_params(labelsize = 15)


plt.xlim(xmin=0)
plt.xlim(xmax=m[2] - n) # 10100)
plt.ylim(ymin=0)
plt.ylim(ymax=maxTime)

plt.show()
