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
m = 64
beta = 1 # 1時間に一回成功
ease = 8 # 10000 倍のマイニング計算能力
lamb = 1  # (2, 6)

# データの作成
n = np.arrange(0, 4000, 10) # np.arange(5000)
num = 4
dist = 64
y = [None] * num
for i in range(num): 
    y[i] = func1(n, m + dist * i, beta, lamb, ease)


# グラフの描画
for i in range(num): 
    string = "m = " + str(i + 1)
    plt.plot(n, y[i], label=string)
plt.xlabel("The number of attacker's transactions")
plt.ylabel("The cumulative distribution probability")

plt.legend()

plt.xlim(xmin=0)
plt.xlim(xmax=10100) # 10100)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)
plt.show()
