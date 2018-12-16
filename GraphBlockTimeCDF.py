# 伝播時間の上限・下限をCDFベースで計算
import scipy.special as spys
import scipy.misc as spym # コンビネーション用
from scipy import stats
import numpy as np
import math
from math import modf
import matplotlib.pyplot as plt
from mpmath import mp
import time as timeKeeper
import pickle

# 時間を計測
startTime = timeKeeper.time()

####  関数定義
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
    return np.exp( calcA(PFork, lamb) * ( 1 - np.log(calcWasteTime(PFork, lamb)) ) )


# 次の逆関数計算に用いる２関数
def myExtendLambert(oneMPFdata):
    return mp.lambertw(oneMPFdata, k=-1).real

def convStr(oneFloat, wasteTime):
    pattern = "%.8fE%.0f"
    return (pattern % (-modf(oneFloat)[0] / wasteTime, modf(oneFloat)[1]))

# https://qiita.com/Ushio/items/0040b3c74a480c46c80c
# mpfを用いて計算, spys.lambertwは非常に小さい値でオーバーフローした
univLambert = np.frompyfunc(mp.lambertw, 2, 1)
all_myExtendLambert = np.frompyfunc(myExtendLambert, 1, 1)
all_float = np.frompyfunc(float, 1, 1)
all_convStr = np.frompyfunc(convStr, 2, 1)
def calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, x):
    # 入力処理
    tmpVal = np.log((1-x)/coeffi) * (1/(lambXlamb * np.log(10))) # x.shape[0]行x.shape[1]列
    tmpVal = tmpVal.reshape(x.size)
    tmpVal = all_convStr(tmpVal, wasteTime)  # x.sizeのリスト
    tmpVal = mp.matrix(tmpVal)
    # tmpVal = np.array(- np.power((1 - x) / coeffi, 1 / lambXlamb) /  wasteTime) # x.shape[0]行x.shape[1]列
    # 出力処理
    retVal = - wasteTime * all_myExtendLambert(tmpVal)
    retVal = np.array(all_float(retVal))
    retVal = retVal.reshape(x.shape[0], x.shape[1])
    # print(retVal)
    return retVal

#### Calculating Upper Bound ####
def calcAndPlotUpperBound(_blockNum, _lamb, _time):
    y = func1(_blockNum, _lamb, _time)
    # グラフの描画
    plt.plot(_time, y, label="Upper Bound", color="red")

#### Ploting Real Data ####
def plotRealData(_blockNum, _eachBlockTS):
    '''
    timeToMineBlocks = [0] * ((len(_eachBlockTS) - 1) // _blockNum)
    for i in range((len(_eachBlockTS) - 1) // _blockNum):
        timeToMineBlocks[i] = _eachBlockTS[(i + 1) * _blockNum] - _eachBlockTS[i * _blockNum]
    '''
    f = open("./NoDelayBlockTimeData.dat","rb")
    dataTS = pickle.load(f)
    f.close()
    timeToMineBlocks = [0] * ((len(dataTS) - 1) // _blockNum)
    for i in range(( len(dataTS) - 1 ) // _blockNum):
        timeToMineBlocks[i] = sum( dataTS[ i * _blockNum : (i + 1)*_blockNum] )

    # plt.hist(xN, bins=50,normed=True, color="#000080", ec='black')
    plt.hist(timeToMineBlocks, bins=1000,normed=True, histtype='step', cumulative=True, label='Simulation Data', color="green")

#### Calculating Lower Bound ####
def calcAndPlotLowerBound(_blockNum, _lamb, _ProbFork, _time):
    # each parameters
    diff = 0.001
    wasteTime = calcWasteTime(_ProbFork, _lamb)
    coeffi = calcC(_ProbFork, _lamb)
    lambXlamb = calcA(_ProbFork, _lamb)

    if (_blockNum == 1):
        Tw = calcWasteTime(_ProbFork, _lamb)
        tgtElements = _time > Tw
        tmpTime = _time[tgtElements]
        print(tmpTime)
        Coeffi = np.exp(_lamb * Tw * (1 - np.log(Tw)) )
        ExpoCoeffi = - _lamb * (tmpTime - Tw * np.log(tmpTime))
        y = 1 -  Coeffi * np.exp(ExpoCoeffi)
        print(len(y))
        print(calcWasteTime(_ProbFork, _lamb))
        # グラフの描画
        plt.plot(tmpTime, y, label="Lower Bound", color="blue")
        return

    # number of times of Monte Carlo
    MCN = 10000 # 100000
    iterate = 20 # 200
    yPlotProb = np.arange(0.08, 0.921, 0.07) # 13 plot
    yPlotProb = np.append([0.01], yPlotProb)
    yPlotProb = np.append(yPlotProb, [0.99])
    print(yPlotProb)
    xPlotTime = np.zeros(iterate * len(yPlotProb)).reshape(iterate, len(yPlotProb))

    # y = [0] * MCN

    for i in range(iterate):
        xVal = np.random.rand(_blockNum, MCN) + 0.0000000000000000000001
        y = np.sum(calcInvFuncCDF(_ProbFork, _lamb, wasteTime, coeffi, lambXlamb, xVal), axis=0) # 行方向に足し合わせる
        print(str(i) + ": loops")
        # print(calcInvFuncCDF(PFork, lamb, wasteTime, coeffi, lambXlamb, np.array([[0.96, 0.95], [0.97,0.99]])))
        ySort = np.sort(y)
        xPlotTime[i] = [ySort[int(MCN * 0.01)],
                        ySort[int(MCN * 0.08)], ySort[int(MCN * 0.15)], ySort[int(MCN * 0.22)], ySort[int(MCN * 0.29)],
                        ySort[int(MCN * 0.36)], ySort[int(MCN * 0.43)], ySort[int(MCN * 0.50)], ySort[int(MCN * 0.57)], ySort[int(MCN * 0.64)],
                        ySort[int(MCN * 0.71)], ySort[int(MCN * 0.78)], ySort[int(MCN * 0.85)], ySort[int(MCN * 0.92)],
                        ySort[int(MCN * 0.99)]]


    var = np.var(xPlotTime, axis=0)
    std = np.std(xPlotTime, axis=0)
    mean = np.mean(xPlotTime, axis=0)

    CI = 0.999
    a = stats.norm.interval(alpha=CI, loc=0, scale=1) # CI: Certification Intervals
    err  = a[1] * std
    ### 0.95の場合の, z * std: 9.25111678933, mean: 2132.29302199[sec]
    print("99.9%の信頼区間(0.99の確率での時間について): " + str(err[len(err) - 1]))
    plt.errorbar(mean,yPlotProb,xerr=err, ecolor='blue',capsize=5, elinewidth=1, markeredgewidth=1)
    plt.plot(mean,yPlotProb, marker="o", markersize=4, markeredgecolor="blue", markerfacecolor="white", color='blue', label="Lower Bound\n(Monte Carlo)")

def funcTime(blockNum, lamb, prob):
        timeForCalc = 100
        incr = 100
        now = 0
        flag = 0
        while (np.abs(now - prob) > 0.001 * 0.0001):
            now = func1(blockNum, lamb, timeForCalc)
            timeForCalc += incr
            if (flag == 0 and now > prob): # 0.005
                flag = 1
                incr = - 1 * incr / 2
            elif (flag == 1 and now < prob):
                flag = 0
                incr = - 1 * incr / 2
        return timeForCalc


###  実行関数
### File Open ###
def main():
    # ファイルの読み込み
    f = open("./BlockTimeData.dat","rb")
    eachBlockTS = pickle.load(f)
    f.close()
    f = open("./forkNum.dat")
    forkBlockNum = 1 # 467
    f.close()

    meanBlockTime = (eachBlockTS[len(eachBlockTS) - 1] - eachBlockTS[0]) / (len(eachBlockTS) - 1)
    ProbFork = forkBlockNum / (len(eachBlockTS) - 1) # 46 / 9999

    # blockNumだけブロックを伸ばすのにかかる時間を評価する
    blockNum = 6
    lamb = 1 / meanBlockTime * (1 + ProbFork) # 1ブロックマイニングするのにかかる平均時間
    print(lamb)

    # 各種パラメータ定義
    stride = 0.4
    start = stride
    end = meanBlockTime * (blockNum * 5.0) # 13000 # 100: 2800, 250: 5500
    time = np.arange(start, end, stride)


    ## For Debug
    print("平均ブロック生成時間: " + str(meanBlockTime) )
    print("ブロックチェーンフォーク確率: " + str(ProbFork))

    ## Calculate Upper Bound CDF
    calcAndPlotUpperBound(blockNum, lamb, time)

    ## Ploting Real Data
    plotRealData(blockNum, eachBlockTS)

    ## Calculate Lower Bound CDF
    calcAndPlotLowerBound(blockNum, lamb, ProbFork, time)


    '''
    ### 確率の時間を決定する関数 ###


    ### 90%のブロックが収まる時間の範囲を計算 ###
    prob = 0.90
    timeForCalc = funcTime(validNum, lamb1, prob)
    print("\n--- No Delay 90% のブロックが入る時間と実際に入っているブロック数 ---")
    numInInterval = sum(tmpX < timeForCalc for tmpX in xN) # tmpX > timeForCalc and
    print(str(timeForCalc) + "[sec]: " + str(100 * numInInterval / len(xN)) + " %")

    '''
    ### 時間の測定結果 ###
    elapsed_time = timeKeeper.time() - startTime
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    ### naiveな方式 ###
    '''
    print("\n--- 指数分布に従うと仮定 ---")
    prob = 0.995
    time1 = funcTime(validNum, 1/meanTime, prob)
    prob = 0.005
    time2 = funcTime(validNum, 1/meanTime, prob)
    print("[" + str(time2) + ", " + str(time1) + "]")
    numInInterval = sum(tmpX < time1 for tmpX in xN) # tmpX > timeForCalc and
    print("0.995: " + str(100 * numInInterval / len(xN)) + " %")
    numInInterval = sum(tmpX < time2 for tmpX in xN) # tmpX > timeForCalc and
    print("0.005: " + str(100 * numInInterval / len(xN)) + " %")

    xData = np.sort(xN)
    '''
    '''
    print("\n--- 実際のデータの時間 ---")
    print(str(xData[10]) + ", " + str(xData[1990]))
    '''
    ### プロットの範囲 ###
    plt.xlim([0, end + 1])
    plt.ylim([0, 1.08])


    ### グラフの調整 ###
    plt.xlabel("Elapsed Time to mine 1 block [sec]", fontsize=20)
    plt.ylabel("CDF", fontsize=20)
    plt.title("Block vs. Time", fontsize=20)
    plt.legend(loc='lower right', fontsize=20) # upper left
    plt.tick_params(labelsize = 15)
    plt.show()

main()
