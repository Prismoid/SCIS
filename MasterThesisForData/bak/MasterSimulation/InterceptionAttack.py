import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# z + zDash = 1.0に注意
def func(lamb, Delay): 
    return 1 - np.exp(- lamb * Delay)

#### パラメータの設定 ####
# 有効ブロックハッシュ(Ethereumが辿れる前のブロックハッシュ数)
m = np.arange(128, 257, 32)
num = len(m)
# 255ブロック先まで作られてしまっていたらもうデータベース更新はない
# 254ブロックまでだったら作成されてても255ブロック目で取り込まれる
# M = m - 1 
# ハッシュレート[block/sec]
lamb = 1 / (3600 * 3) # 3時間に一回とする3600 * 3
lamb_a = lamb * 100 # attacker 100倍のハッシュレート

# データの作成
# 遅延時間
stride = 0.2
start = stride
end = 120
Delay = np.arange(start, end, stride)
y = [None] * num
print(y)
print(m[3])
for i in range(num): 
    y[i] = func(lamb_a, Delay)

print(Delay)
print(lamb)
print(lamb * Delay)

# グラフの描画
for i in range(num):
    string = "m = " + str(m[i])
    plt.plot(Delay, y[i], label=string)
    

plt.xlabel("The ready time for processing Tx(D_all(h))")
plt.ylabel("The probability of failing to get ID Space")

plt.legend()

plt.xlim(xmin=0)
plt.xlim(xmax=end) # 10100)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)
plt.show()

