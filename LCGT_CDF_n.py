### package
import scipy.special as spys
import scipy.misc as spym # コンビネーション用
from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt
from mpmath import mp
from math import modf

#### 関数定義
def func1(m, lamb, time):
    coeffi = np.power(math.exp(1), (- lamb * time))
    ret_sum = 0;
    if (m > 1):
        for i in range(m):
            tmp_log = np.power(lamb * time, i) / math.factorial(i)
            ret_sum += tmp_log
            '''
            tmp_log =  i * np.log(lamb * time) - math.lgamma(i + 1)
            ret_sum += np.power(math.exp(1), tmp_log)
            '''
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
    return np.exp( calcA(PFork, lamb) * ( 1 - np.log(calcWasteTime(PFork, lamb) ) ) )

# 次の逆関数計算に用いる２関数
def myExtendLambert(oneMPFdata):
    return mp.lambertw(oneMPFdata, k=-1).real

def convStr(oneFloat, wasteTime):
    pattern = "%.8fE%.0f"
    return (pattern % (-modf(oneFloat)[0] / wasteTime, modf(oneFloat)[1]))
def calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, x):
    return - wasteTime * spys.lambertw(- np.power((1 - x) / coeffi, 1 / lambXlamb) /  wasteTime, -1).real
def tmpFunc(PFork, lamb, wasteTime, coeffi, lambXlamb, x):
    return - np.power((1 - x) / coeffi, 1 / lambXlamb) /  wasteTime
# https://qiita.com/Ushio/items/0040b3c74a480c46c80c
# mpfを用いて計算, spys.lambertwは非常に小さい値でオーバーフローした
univLambert = np.frompyfunc(mp.lambertw, 2, 1)
all_myExtendLambert = np.frompyfunc(myExtendLambert, 1, 1)
all_float = np.frompyfunc(float, 1, 1)
all_convStr = np.frompyfunc(convStr, 2, 1)
def calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, x):
    # 入力処理
    tmpVal = np.log((1-x)/coeffi) * (1/(lambXlamb * np.log(10))) # x.shape[0]x.shape[1]列
    tmpVal = tmpVal.reshape(x.size)
    tmpVal = all_convStr(tmpVal, wasteTime)  # x.sizeのリスト
    tmpVal = mp.matrix(tmpVal)
    # tmpVal = np.array(- np.power((1 - x) / coeffi, 1 / lambXlamb) /  wasteTime) # x.shape[0]行x.shape[1]列
    # 出力処理
    retVal = - wasteTime * all_myExtendLambert(tmpVal)
    retVal = np.array(all_float(retVal))
    retVal = retVal.reshape(x.shape[0], x.shape[1])
    return retVal

## execute MCN
def executeMCN(plt, PFork, sampleNum, iterateNum, colorName, seedNum):
    np.random.seed(seedNum)
    # each parameters
    wasteTime = calcWasteTime(PFork, lamb)
    coeffi = calcC(PFork, lamb)
    lambXlamb = calcA(PFork, lamb)
    # number of times of Monte Carlo
    MCN = sampleNum # 100000
    iterate = iterateNum # 200
    yPlotProb = np.arange(0.1, 0.901, 0.1) # 9 plot
    xPlotTime = np.zeros(iterate * len(yPlotProb)).reshape(iterate, len(yPlotProb))
    for i in range(iterate):
        print(i)
        xVal = np.random.rand(blockNum, MCN)
        y = np.sum(calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, xVal), axis=0) # 行方向に足し合わせる
        ySort = np.sort(y)
        xPlotTime[i] = [ySort[int(MCN * 0.1)], ySort[int(MCN * 0.2)],
                        ySort[int(MCN * 0.3)], ySort[int(MCN * 0.4)],
                        ySort[int(MCN * 0.5)], ySort[int(MCN * 0.6)],
                        ySort[int(MCN * 0.7)], ySort[int(MCN * 0.8)],
                        ySort[int(MCN * 0.9)]]
    # 各種統計情報
    var = np.var(xPlotTime, axis=0)
    std = np.std(xPlotTime, axis=0)
    mean = np.mean(xPlotTime, axis=0)
    # 信頼区間
    CI = 0.999
    a = stats.norm.interval(alpha=CI, loc=0, scale=1) # CI: Certification Intervals
    err  = a[1] * std
    print(err)
    # 0.95の場合の, z * std: 9.25111678933, mean: 2132.29302199[sec]
    print("99.9%の信頼区間(0.99の確率での時間について): " + str(err[len(err) - 1]))
    plt.errorbar(mean,yPlotProb,linestyle="--", xerr=err, color=colorName,capsize=5, elinewidth=1, markeredgewidth=1)
    PForkStr = "{:0<4}".format(PFork)
    labelName = "Asynchronized, $P_F = " + PForkStr + "$ (Lower Bound)"
    plt.plot(mean,yPlotProb, "--", marker="o", markersize=4, markeredgecolor=colorName, markerfacecolor="white", color=colorName, label=labelName)

# データからプロット
def executeMCNfromData(plt, xPlotTime, PFork, colorName):
    # 各種統計情報
    var = np.var(xPlotTime, axis=0)
    std = np.std(xPlotTime, axis=0)
    mean = np.mean(xPlotTime, axis=0)
    # plot on graph
    yPlotProb = np.arange(0.1, 0.901, 0.1) # 9 plot
    PForkStr = "{:0<4}".format(PFork)
    labelName = "Asynchronized, $P_F = " + PForkStr + "$ (Lower Bound)"
    plt.plot(mean,yPlotProb, "--", marker="o", markersize=4, markeredgecolor=colorName, markerfacecolor="white", color=colorName, label=labelName)


#### 関数定義ここまで

### --- Plotting Graph --- ###
# blockchain parameters
blockNum = 6
lamb = 1 / 600
# time data
stride = 0.1
start = stride
end = 10000 # 100: 2800, 250: 5500
time = np.arange(start, end, stride)

########## Synchronized Network, $P_F = 0.00$
y1 = func1(blockNum, lamb, time)
# グラフの描画
plt.plot(time, y1, label="Synchronized,  $P_F = 0.00$", color="red")

######### Asynchronized, $P_F = 0.169$ (Lower Bound)
PFork = 0.1
executeMCN(plt, PFork, 5000, 50, "green", 2)
PFork = 0.05
executeMCN(plt, PFork, 5000, 50, "blue", 2)
######### Asynchronized from data
'''
PFork = 0.1
executeMCNfromData()
'''
### プロットの範囲 ###
plt.xlim([0, end + 1])
plt.ylim([0, 1.08])

### グラフの調整 ###
plt.xlabel("Time for the longest chain to grow by 6 blocks [sec]", fontsize=20)
plt.ylabel("CDF", fontsize=20)
# plt.title("Block vs. Time", fontsize=20)
plt.legend(loc='lower right', fontsize=18) # upper left
plt.tick_params(labelsize = 15)
plt.show()
