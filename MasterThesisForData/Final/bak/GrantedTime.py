import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# m - l - nブロック以上のブロックが生成される確率
def func(m, n, l, lamb, Delay): 
    coeffi = np.power(math.exp(1), (- lamb * Delay))
    ret_sum = 0;
    for i in range(m - n - l + 1):
        tmp_log =  i * np.log(lamb * Delay) - math.lgamma(i + 1)
        ret_sum += np.power(math.exp(1), tmp_log)
    return  coeffi * ret_sum

#### パラメータの設定 ####
# ハッシュレート[block/sec]
lamb = 1 / 14.31 * (1 + 0.06192)

# 有効ブロックハッシュ(Ethereumが辿れる前のブロックハッシュ数)
n = 100
l    = [None] * 3
time = [None] * 3
l[0] = np.arange(10, 2 * n, 10)
l[1] = np.arange(10, 3 * n, 10)
l[2] = np.arange(10, 4 * n, 10)
time[0] = [None] * len(l[0])
time[1] = [None] * len(l[1])
time[2] = [None] * len(l[2])

# データの作成
for j in range(3): 
    for i in range(len(l[j])):
        now = 0
        timeForCalc = 1
        incr = 100
        flag = 0
        while (np.abs(now - 0.001) > 0.001 * 0.0001):
            now = func(n * (j + 3), n, l[j][i], lamb, timeForCalc)
            #        print(now)
            timeForCalc += incr
            if (flag == 0 and now < 0.001):
                flag = 1
                incr = - 1 * incr / 2
            elif (flag == 1 and now > 0.001):
                flag = 0
                incr = - 1 * incr / 2
        time[j][i] = timeForCalc

print(time[0])
print(time[1])
print(time[2])
# グラフの描画
plt.plot(l[0], time[0], label="m = n * 3")
plt.plot(l[1], time[1], label="m = n * 4")
plt.plot(l[2], time[2], label="m = n * 5")

plt.xlabel("Needed blocks to include a registaration transaction", fontsize=20)
plt.ylabel("Granted delay time [sec]", fontsize=20)

plt.xlim(xmin=0)
plt.xlim(xmax=450) # 10100)
#plt.ylim(ymin=0.0)
#plt.ylim(ymax=1.05)

plt.legend(loc='upper right', fontsize=20)
plt.tick_params(labelsize = 15)

plt.show()
