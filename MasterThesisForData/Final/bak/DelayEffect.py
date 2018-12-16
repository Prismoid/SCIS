import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# m - l - nブロック以下しかブロックが生成されない確率
def func(m, l, n, lamb, Delay): 
    coeffi = np.power(math.exp(1), (- lamb * Delay))
    ret_sum = 0;
    for i in range(m - l - n):
        tmp_log =  i * np.log(lamb * Delay) - math.lgamma(i + 1)
        ret_sum += np.power(math.exp(1), tmp_log)
    return  coeffi * ret_sum

def func2(n, l2, lamb, Delay):
    coeffi = np.power(math.exp(1), (- lamb * Delay))
    ret_sum = 0;
    for i in range(n + 1 - l2):
        tmp_log =  i * np.log(lamb * Delay) - math.lgamma(i + 1)
        ret_sum += np.power(math.exp(1), tmp_log)
    return  coeffi * ret_sum

#### パラメータの設定 ####
PFork = 0.06192
lamb = 1 / 14.31 * (1 + PFork)

# データの作成
# 遅延時間
D = 500
n = 250 
m = 250 * 3
l = np.arange(1,m - n,1)
l2 = np.arange(1,n,1)
D2 = 500

y = [None] * len(l)
y2 = [None] * len(l2)
# print(m[3])
for i in range(len(l)): 
    y[i] = func(m, l[i], n, lamb, D)
for i in range(len(l2)):
    y2[i] = func2(n, l2[i], lamb, D2)
plt.xlabel("l")
plt.ylabel("Probability ")

plt.plot(l, y, c='r', label="Analysis Result(1)")
plt.plot(l2, y2, c='b', label="Analysis Result(2)")
plt.legend()

plt.show()
