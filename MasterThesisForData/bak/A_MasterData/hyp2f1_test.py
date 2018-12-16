import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# z + zDash = 1.0に注意
def func(k, m, beta, lamb): 
    tmp_k = beta / (beta + lamb)
    tmp_m = lamb / (beta + lamb)
    return np.power(math.exp(1), k * np.log(tmp_k) + m * np.log(tmp_m) + math.lgamma(k + m) - math.lgamma(k + 1) - math.lgamma(m))
    # return spy.hyp2f1(1, m + n + 1, n + 2, 1000/1001)

def funcSum(n, m, beta, lamb):
    retSum = 0
    for k in range(n):
        retSum += func(k, m, beta, lamb)
    return retSum

# パラメータの設定
m   = 64
maxNum = 180 # maximum number of attacker's transaction
beta = 1 / 3600 * 100 # 1時間に1回成功, 攻撃者はその200倍の計算資源
lamb = 1 / 14.31  # (2, 6)

# データの作成
retSum = 0
interval = 1
n = np.arange(0, maxNum + interval, interval)
interNum = int(maxNum / interval + 1)
print(interNum)
plotNum = 4
dist    = 64
y = np.zeros((plotNum, interNum))
for i in range(plotNum):
    for j in range(interNum):
        if j != 0 : 
            y[i][j] = func(j, m + dist * i, beta, lamb) + y[i][j - 1]
        else :
            y[i][j] = func(j, m + dist * i, beta, lamb)


# グラフの描画
for i in range(plotNum): 
    string = "m = " + str(m + i * dist)
    plt.plot(n, y[i], label=string)
plt.xlabel("The max number of transactions which can be created by an attacker\n")
plt.ylabel("Probability")

plt.legend()

plt.xlim(xmin=0)
plt.xlim(xmax=maxNum + 1) # 10100)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)
plt.show()
