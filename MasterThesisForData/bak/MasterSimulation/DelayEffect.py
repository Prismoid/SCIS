import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# z + zDash = 1.0に注意
def func(m, l, lamb, Delay): 
    coeffi = np.power(math.exp(1), (- lamb * Delay))
    ret_sum = 0;
    for i in range(m - l):
        tmp_log =  i * np.log(lamb * Delay) - math.lgamma(i + 1)
        ret_sum += np.power(math.exp(1), tmp_log)
    return  coeffi * ret_sum

#### パラメータの設定 ####
# 有効ブロックハッシュ(Ethereumが辿れる前のブロックハッシュ数)
m = np.arange(128, 257, 64)
num = len(m)
l = 12 # confirmation block
# 255ブロック先まで作られてしまっていたらもうデータベース更新はない
# 254ブロックまでだったら作成されてても255ブロック目で取り込まれる
# M = m - 1 
# ハッシュレート[block/sec]
lamb = 1 / 11.6189495774

# データの作成
# 遅延時間
stride = 0.2
start = stride
end = 2201
Delay = np.arange(start, end, stride)
y = [None] * num
print(y)
# print(m[3])
for i in range(num): 
    y[i] = func(m[i], l, lamb, Delay * 2)

print(Delay)
print(lamb)
print(lamb * Delay)

# グラフの描画
for i in range(num):
    # if(i == num - 1): 
    string = "m = " + str(m[i]) + " + 1 (Upper Bound)"
    plt.plot(Delay, 1.0 - y[i], label=string)
    

plt.xlabel("The delay time's upper bound (D)")
plt.ylabel("The probability of failing to issue ID")

plt.xlim(xmin=0)
plt.xlim(xmax=end + 100) # 10100)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)

start = 1100
end = 2000
stride = 100
delay = np.arange(start, end, stride) 
result = [0.0, 0.0, 0.0002, 0.0078, 0.0729, 0.3278, 0.6875, 0.9191, 0.9895]
plt.scatter(delay, result, c='r', label="Simulation Result(m = 256 + 1)")
plt.legend()

plt.show()
