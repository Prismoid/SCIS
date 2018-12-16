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
m = np.arange(128, 257, 32)
num = len(m)
l = 12 # confirmation block
# 255ブロック先まで作られてしまっていたらもうデータベース更新はない
# 254ブロックまでだったら作成されてても255ブロック目で取り込まれる
# M = m - 1 
# ハッシュレート[block/sec]
lamb = 1 / 14.31

# データの作成
# 遅延時間
stride = 0.2
start = stride
end = 3401
Delay = np.arange(start, end, stride)
y = [None] * num
print(y)
print(m[3])
for i in range(num): 
    y[i] = func(m[i], l, lamb, Delay * 2)

print(Delay)
print(lamb)
print(lamb * Delay)

# グラフの描画
for i in range(num):
    string = "m = " + str(m[i]) + " + 1 (Upper Bound)"
    plt.plot(Delay, 1.0 - y[i], label=string)
    

plt.xlabel("The delay time's upper bound (D)")
plt.ylabel("The probability of failing to issue ID")

plt.xlim(xmin=0)
plt.xlim(xmax=end + 100) # 10100)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)

start = 1500
end = 3401
stride = 100
delay = np.arange(start, end, stride) 
result = [0.0001, 0.0025, 0.0172, 0.0671, 0.16, 0.2744, 0.4034, 0.5068, 0.591, 0.668, 0.7321, 0.7804, 0.8302, 0.863, 0.8924, 0.9117, 0.9318, 0.9463, 0.9619, 0.9688]
plt.scatter(delay, result, c='r', label="Simulation Result(m = 256 + 1)")
plt.legend()

plt.show()
