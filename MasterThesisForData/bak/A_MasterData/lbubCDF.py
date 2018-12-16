import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# z + zDash = 1.0に注意
def func1(m, lamb, time): 
    coeffi = np.power(math.exp(1), (- lamb * time))
    ret_sum = 0;
    for i in range(m - 1):
        tmp_log =  i * np.log(lamb * time) - math.lgamma(i + 1)
        ret_sum += np.power(math.exp(1), tmp_log)
    return  1 - coeffi * ret_sum

def func2(m, a, Xlamb, lamb, time):
    Const = m * Xlamb / (1 - a)
    if (time <= Const + 0.001):
        return 0
    time2 = time - Const
    coeffi = np.power(math.exp(1), (- lamb * time2))
    ret_sum = 0;
    for i in range(m - 1):
        tmp_log =  i * np.log(lamb * time2) - math.lgamma(i + 1)
        ret_sum += np.power(math.exp(1), tmp_log)
    return  1 - coeffi * ret_sum

#### パラメータの設定 ####
# 有効ブロックハッシュ(Ethereumが辿れる前のブロックハッシュ数)
validNum = 100
lamb1 = 1 / 11.6189495774
lamb3 = 1 / 14.31
Xlamb = 0.918845697578
a = (1 - np.sqrt(Xlamb * 1 / 14.31)) / (1 - (Xlamb * 1 / 14.31))
lamb2 = 1 / 14.31 * a

print(a)
print(str(1/lamb2))

# データの作成
# 遅延時間
stride = 0.4
start = stride
end = 3000
time = np.arange(start, end, stride)
y1 = func1(validNum, lamb1, time)
y2 = [None] * len(time)
for i in range(len(time)): 
    y2[i] = func2(validNum, a, Xlamb, lamb2, time[i])
y4 = [None] * len(time)
for i in range(len(time)):
    y4[i] = func2(validNum, 0.7, Xlamb, lamb3 * 0.7, time[i])
y3 = func1(validNum, lamb3, time)

# グラフの描画
plt.plot(time, y1, label="Upper Bound")
plt.plot(time, y2, label="Lower Bound")
plt.plot(time, y3, label="example")
plt.plot(time, y4, label="XX")
    

plt.xlabel("The delay time's upper bound (D)")
plt.ylabel("The probability of failing to issue ID")

plt.xlim(xmin=0)
plt.xlim(xmax=end + 100) # 10100)
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)

plt.legend()

plt.show()
