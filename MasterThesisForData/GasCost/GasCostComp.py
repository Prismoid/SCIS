# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# cf. https://tonton7.blogspot.jp/2016/09/matplotlib.html#005
# cf. http://yubais.net/doc/matplotlib/bar.html

def main():
    incrReg = 105745
    powReg  = 85548
    preReg1 = 40886
    preReg2 = 85932
    txGas   = 21000
    # data
    y1 = [preReg1 + preReg2 + 2 * txGas, incrReg + txGas, powReg + txGas]
    y2 = [161000, 121000, 101000]
    x = np.arange(len(y1))
    
    x = np.arange(len(y1)) + 1
    X = 0
    w = 0.4


    plt.ylim(ymax=180000)
    
    plt.bar(x, y1, align="center", color="b", label="GasEstimate", width=w)           # 中央寄せで棒グラフ作成
    plt.bar(x + w, y2, align="center", color="g", label="SSTORE and Tx Cost", width=w)
    
    plt.xticks(x, ["pre-order","increment","PoW"], size=16)  # X軸のラベル

    # 線を入れる
    plt.plot([0.81, 1.19],[61886, 61886], "red")
    plt.plot([1.21, 1.59],[61000, 61000], "red")
    

    plt.legend(loc="best")
    ax = plt.gca()
    ax.yaxis.grid(color="black")
    plt.yticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000], size="12")
    plt.ylabel("Gas Cost (per Tx))", size=16)
    plt.xticks(x + w/2, ['pre-order','increment','PoW'])
    plt.show()


if __name__ == '__main__':
    main()
