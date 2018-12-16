import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt

# PDF, RumorSpreading
def func(beta, lamb, a, t):
    if (beta * t < 700): 
        return (lamb * a * np.power(math.exp(1), beta * t)) * (np.power(1 / (1 - a + a * np.power(math.exp(1), beta * t)), lamb/beta + 1))
    else:
        return (lamb * a) * np.power(a, -lamb / beta - 1) * np.power(np.e, -lamb * t)

def funcCDF(beta, lamb, a, t):
    if (beta * t < 700): 
        return 1.0 - np.power(1 - a + a * np.power(np.e, beta * t), -lamb / beta)
    else:
        return 1.0 - np.power(a, -lamb / beta) * np.power(np.e, -lamb * t)

def funcExpect(beta, lamb, a, t):
    if (beta * t < 700): 
        return np.power(1 - a, -lamb / beta) * np.power(1 + a/(1-a) * np.power(np.e, beta * t), -lamb / beta)
    else:
        return np.power(a, -lamb / beta) * np.power(np.e, -lamb * t)
    
def expDist(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

def expDistCDF(t, lamb):
    return 1.0 - np.power(np.e, - lamb * t)




# パラメータの設定
beta = 10.2261233386 # 9.13699302846
lamb = 0.0741966666667 # 1 / 14.3098
a = 1 / 5000 # 5000ノード
diff = 0.1

#### 一致部分の確率を計算 ####
retSum = 0
kizami = 0.000001

'''
# 10.8500886926(beta=10, lamb=0.1, a=1/5000, kizami=0.000001), 10.8500870501 <- python3の超幾何関数を利用
print(funcExpect(beta, lamb, a, kizami * 2000000))
for i in range(1000000000):
    # if (funcExpect(beta, lamb, a, kizami * (i + 500000)) == 0.0):
      #  print(i)
    retSum += funcExpect(beta, lamb, a, kizami * i) * kizami
print(retSum)
#### ここまで ####
'''

# データの作成
maxT = 100
t = np.arange(0, maxT, diff)
y = [None] * len(t)
for i in range(len(t)): 
    y[i] = func(beta, lamb, a, i * diff) # for PDF
# print(y)

yExp = expDist(t, lamb)# =1/14.31)

plt.plot(t, yExp, color="#ff0000", label="Synchronized Network")

for i in range(500):
    if ((lamb * a * np.power(math.exp(1), beta * t[i]) / beta) * (np.power(1 / (1 - a + a * np.power(math.exp(1), beta * t[i])), lamb/beta + 1)) < 0):
        print("HE?")
        
# グラフの描画
# CDF
'''
plt.plot(t, y, label="Data")
plt.xlabel("Time")
plt.ylabel("CDF")

plt.legend()

plt.xlim(xmin=0)
plt.xlim(xmax=maxT + 1) 
plt.ylim(ymin=0.0)
plt.ylim(ymax=1.05)
plt.show()
'''
# PDF
plt.plot(t, y, label="Asynchronized Network")
plt.xlabel("Block Time [sec]")
plt.ylabel("PDF")

plt.legend()

# plt.xlim(xmin=0)
# plt.xlim(xmax=maxT + 1) 
# plt.ylim(ymin=0.0)
# plt.ylim(ymax=1.05)
plt.show()
