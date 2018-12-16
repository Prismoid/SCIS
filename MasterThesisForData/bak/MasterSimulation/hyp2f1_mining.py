import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# z + zDash = 1.0に注意
def func1(n, m, beta, lamb, ease): 
    z = beta * ease / (beta * ease + lamb)
    zDash = lamb / (beta * ease + lamb)
    return 1 -  np.power(zDash, m) * spys.hyp2f1(1, m + n + 1, n + 2, z) * spym.comb(m + n, m - 1) * np.power(z, n + 1)
    # return spy.hyp2f1(1, m + n + 1, n + 2, 1000/1001)

# パラメータの設定
m = 1
beta = 1
lamb = np.power(2,24) # Ethereumの例: 72661GH/s, 30MH/s, 2^26=67108864, 2422033
lamb = 242203300
ease = np.power(2,16)

# データの作成
# m = np.arange(0, 1000000, 100000) # np.arange(5000)
m = np.arange(2000)
y = [0, 1, 2]

# 実際に近いデータ
reallamb = 2422033
realy = 1.0 - func1(0, m, beta, reallamb, ease)
# 実際にマイニングした場合のデータ
reallamb2 = 2422033
realy2 = 1.0 - func1(0, m, beta, reallamb2, 1.0)

lambs = [np.power(2, 20), np.power(2, 22), np.power(2, 24)]
num = 3
for i in range(num): 
    y[i] = 1.0 - func1(0, m, beta, lambs[i], ease)

# グラフの描画
for i in range(num): 
    string = "PoW Tx maker's resource ratio: 1 / " + str(2) + "^" + str(20 + i * 2)
    plt.plot(m, y[i], label = string)
plt.plot(m, realy, label="Real Example(Ethereum): 1 / 2422033")
plt.plot(m, realy2, label="test")

plt.xlabel("The number of blocks")
plt.ylabel("The probability of getting ID space")

plt.legend()

plt.xlim(xmin=0)
plt.xlim(xmax=2100) # 10100)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)
plt.show()
