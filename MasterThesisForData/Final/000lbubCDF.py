import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

from scipy import stats
import numpy as np
import math 
import matplotlib.pyplot as plt

####  関数定義
def func1(m, lamb, time): 
    coeffi = np.power(math.exp(1), (- lamb * time))
    ret_sum = 0;
    if (m > 1):
        for i in range(m - 1):
            tmp_log =  i * np.log(lamb * time) - math.lgamma(i + 1)
            ret_sum += np.power(math.exp(1), tmp_log)
        return  1 - coeffi * ret_sum
    else:
        return 1 - coeffi

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

def calcUnitFindBlockProb(lamb):
    return 1 - np.exp(- lamb)

def calcWasteTime(PFork, lamb):
    return np.log(1 - PFork) / np.log(1 - calcUnitFindBlockProb(lamb))

def calcA(PFork, lamb):
    return lamb * calcWasteTime(PFork, lamb)

def calcC(PFork, lamb):
    return np.exp( calcA(PFork, lamb) * ( 1 - np.log(calcWasteTime(PFork, lamb)) ) )

def calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, x):
    return - wasteTime * spys.lambertw(- np.power((1 - x) / coeffi, 1 / lambXlamb) /  wasteTime, -1).real

#### 関数定義ここまで

### File Open ###
data = np.loadtxt('z000tsData4100000-4600000.dat')
uncleNum = np.loadtxt('z000uncleData4100000-4600000.dat')
meanLamb = (data[500000] - data[0]) / 500000
dataPFork = uncleNum / 500000

print("平均ブロック生成時間: " + str(meanLamb))
print("ブロックチェーンフォーク確率: " + str(dataPFork))

#### 各種データの設定 ####
### Upper Bound
# 時間データの作成
stride = 0.4
start = stride
end = 5500 # 100: 2800, 250: 5500
time = np.arange(start, end, stride)

# pre-orderが有効になるまでのブロック数
validNum = 250
PFork = dataPFork
lamb1  = 1/meanLamb * (1 + PFork)
y1 = func1(validNum, lamb1, time)

print("Block Time (mean): " + str(meanLamb))
print("Probability of Fork: " + str(PFork))

# グラフの描画
plt.plot(time, y1, label="Upper Bound", color="red")

### Real Data 
x = data
xN = [0] * (500000 // validNum)
for i in range((500000 // validNum)):
    xN[i] = x[(i + 1) * validNum] - x[i * validNum]

# plt.hist(xN, bins=50,normed=True, color="#000080", ec='black')
plt.hist(xN, bins=1000,normed=True, histtype='step', cumulative=True, label='Ethereum Classic \nBC Data', color="green")

### Lower Bound(Monte Carlo)
# input values
PFork = dataPFork
lamb  = 1/meanLamb * (1 + PFork)

# each parameters
diff = 0.001
wasteTime = calcWasteTime(PFork, lamb)
coeffi = calcC(PFork, lamb)
lambXlamb = calcA(PFork, lamb)

# number of times of Monte Carlo
MCN = 100 # 100000
iterate = 200 # 200
yPlotProb = np.arange(0.08, 0.921, 0.07) # 15 plot
yPlotProb = np.append([0.0005, 0.005, 0.025], yPlotProb)
yPlotProb = np.append(yPlotProb, [0.975, 0.995, 0.9995])
print(yPlotProb)
xPlotTime = np.zeros(iterate * len(yPlotProb)).reshape(iterate, len(yPlotProb))
# y = [0] * MCN

for i in range(iterate): 
    xVal = np.random.rand(validNum, MCN)
    y = np.sum(calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, xVal), axis=0) # 行方向に足し合わせる
    ySort = np.sort(y)
    xPlotTime[i] = [ySort[int(MCN * 0.0005)], ySort[int(MCN * 0.005)], ySort[int(MCN * 0.025)], 
                    ySort[int(MCN * 0.08)], ySort[int(MCN * 0.15)], ySort[int(MCN * 0.22)], ySort[int(MCN * 0.29)],
                    ySort[int(MCN * 0.36)], ySort[int(MCN * 0.43)], ySort[int(MCN * 0.50)], ySort[int(MCN * 0.57)], ySort[int(MCN * 0.64)],
                    ySort[int(MCN * 0.71)], ySort[int(MCN * 0.78)], ySort[int(MCN * 0.85)], ySort[int(MCN * 0.92)],
                    ySort[int(MCN * 0.975)],ySort[int(MCN * 0.995)], ySort[int(MCN * 0.9995)]]
'''
mean = np.array([ 1237.69218582,  1260.12029328,  1350.25044321, 1465.32279322,  1513.38893135, 1548.53880716,  1578.1509271,  1604.83752971, 1630.08851102, 1654.83450879,
                  1679.84676754,  1705.96676299,  1734.23406795, 1766.37641797,  1805.82831852, 1861.82790865, 2007.44924415,  2134.59060544,  2168.69103366])
std = np.array([ 4.32312779,  3.21024985,  1.44867482,  0.77764723,  0.64259442,  0.58629381, 0.54733799, 0.55856328, 0.55244524, 0.57064755, 0.57764861, 0.59951346,
        0.6496496, 0.72715041, 0.81514208, 0.90223982, 1.88744663, 4.92012715, 6.90373712])
'''

var = np.var(xPlotTime, axis=0)
std = np.std(xPlotTime, axis=0)
mean = np.mean(xPlotTime, axis=0)
# print(var)
# print(std)
# print(mean)

CI = 0.999
a = stats.norm.interval(alpha=CI, loc=0, scale=1) # CI: Certification Intervals
err  = a[1] * std
print(err)
### 0.95の場合の, z * std: 9.25111678933, mean: 2132.29302199[sec]
print("99.9%の信頼区間(0.99の確率での時間について): " + str(err[len(err) - 1]))
plt.errorbar(mean,yPlotProb,xerr=err, ecolor='blue',capsize=5, elinewidth=1, markeredgewidth=1)
plt.plot(mean,yPlotProb, marker="o", markersize=4, markeredgecolor="blue", markerfacecolor="white", color='blue', label="Lower Bound\n(Monte Carlo)")
# plt.hist(y, bins=100, normed=True, cumulative=True, histtype='step', label='Lower Bound \n (Montecarlo)') #


### 確率の時間を決定する関数 ###
def funcTime(validNum, lamb, prob):
    timeForCalc = 100
    incr = 100
    now = 0
    flag = 0
    while (np.abs(now - prob) > 0.001 * 0.0001):
        now = func1(validNum, lamb, timeForCalc)
        timeForCalc += incr
        if (flag == 0 and now > prob): # 0.005
            flag = 1
            incr = - 1 * incr / 2
        elif (flag == 1 and now < prob):
            flag = 0
            incr = - 1 * incr / 2
    return timeForCalc

### 99%のブロックが収まる時間の範囲を計算 ###
prob = 0.005
timeForCalc = funcTime(validNum, lamb1, prob)
print("\n--- 上限と下限を用いた最低でも99%のブロックが入る時間の範囲 ---")
print("[" + str(timeForCalc) + ", " + str(mean[16]) + "]") # err[17] 信頼区間のパラメータ
print("\n--- 上記値の範囲に実際に入っている % を表示 ---")
numInInterval = sum(tmpX < mean[16] for tmpX in xN) # tmpX > timeForCalc and
print("0.995: " + str(100 * numInInterval / len(xN)) + " %")
numInInterval = sum(tmpX < timeForCalc for tmpX in xN) # tmpX > timeForCalc and
print("0.005: " + str(100 * numInInterval / len(xN)) + " %")

### naiveな方式 ###
print("\n--- 指数分布に従うと仮定 ---")
prob = 0.995
time1 = funcTime(validNum, 1/meanLamb, prob)
prob = 0.005
time2 = funcTime(validNum, 1/meanLamb, prob)
print("[" + str(time2) + ", " + str(time1) + "]")
numInInterval = sum(tmpX < time1 for tmpX in xN) # tmpX > timeForCalc and
print("0.995: " + str(100 * numInInterval / len(xN)) + " %")
numInInterval = sum(tmpX < time2 for tmpX in xN) # tmpX > timeForCalc and
print("0.005: " + str(100 * numInInterval / len(xN)) + " %")

xData = np.sort(xN)
print("\n--- 実際のデータの時間 ---")
print(str(xData[10]) + ", " + str(xData[1990]))
        
### プロットの範囲 ### 
plt.xlim([0, end + 1])
plt.ylim([0, 1.08])

### グラフの調整 ### 
plt.xlabel("Elapsed Time to mine 100 blocks [sec]", fontsize=20)
plt.ylabel("CDF", fontsize=20)
plt.title("Block vs. Time", fontsize=20)
plt.legend(loc='upper left', fontsize=20)
plt.tick_params(labelsize = 15)
plt.show()


### 各データ ###
# Block Height: 4100000-4600000
# ***  validNum = 100 ***
# --- 上限と下限を用いた最低でも99%のブロックが入る時間の範囲 ---
# (99.9% 信頼区間の最大値を用いた)
# [1014.208984375, 2045.70913548]
# 
# --- 上記値の範囲に実際に入っている % を表示 ---
# 99.94 %
# ***  validNum = 250 ***
# --- 上限と下限を用いた最低でも99%のブロックが入る時間の範囲 ---
# (99.9% 信頼区間の最大値を用いた)
# [2833.447265625, 4741.54806728]
#
# --- 上記値の範囲に実際に入っている % を表示 ---
# 99.9 %

# Block Height: 4600000-5100000
# *** validNum = 100 ***
# --- 上限と下限を用いた最低でも99%のブロックが入る時間の範囲 ---
# (99.9% 信頼区間の最大値を用いた)
# [1013.409423828125, 2044.80160993]
# 
# --- 上記値の範囲に実際に入っている % を表示 ---
# 99.88 %
# *** validNum = 250 ***
# --- 上限と下限を用いた最低でも99%のブロックが入る時間の範囲 ---
# (99.9% 信頼区間の最大値を用いた)
# [2831.2744140625, 4740.20114152]
#
# --- 上記値の範囲に実際に入っている % を表示 ---
# 99.95 %
