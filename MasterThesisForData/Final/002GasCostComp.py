# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# cf. https://tonton7.blogspot.jp/2016/09/matplotlib.html#005
# cf. http://yubais.net/doc/matplotlib/bar.html

def main():
    incrReg = 106005
    preReg1 = 41085
    preReg2 = 86975
    txGas   = 21000
    # data
    preOrder = preReg1 + preReg2 + 2 * txGas
    increment = incrReg + txGas
    y1 = [preOrder, increment]
    y2 = [161000, 121000]
    x = np.arange(len(y1))
    
    x = np.arange(len(y1)) + 1
    X = 0
    w = 0.4

    improved = (preOrder - increment) / preOrder * 100
    tImproved = 40000 / 161000 * 100
    print("Improved Gas Cost percentage: " + str(improved) + " %")
    print("Improved Gas Cost percentage(theory): " + str(tImproved) + " %")

    plt.ylim(ymax=180000)
    
    plt.bar(x, y1, align="center", color="b", label="GasEstimate", width=w)           # 中央寄せで棒グラフ作成
    plt.bar(x + w, y2, align="center", color="g", label="SSTORE and Tx Cost", width=w)
    
    plt.xticks(x, ["pre-order","increment"], fontsize=20)  # X軸のラベル

    # 線を入れる
    plt.plot([0.81, 1.19],[61786, 61986], "red") # 61886
    plt.plot([1.21, 1.59],[60900, 61100], "red") # 61000
    

    plt.legend(loc="best", fontsize=20)
    ax = plt.gca()
    ax.yaxis.grid(color="black")
    plt.yticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000], size="20")
    plt.ylabel("Gas Cost", size=20)
    plt.xticks(x + w/2, ['pre-order & registration','increment'], size="20")
    plt.show()


if __name__ == '__main__':
    main()
