import scipy.special as spys
import scipy.misc as spym # コンビネーション用 

import numpy as np
import math 
import matplotlib.pyplot as plt
import sys

# PDF, RumorSpreading
def func(a, beta, t):
    c1 = a/(1-a)
    return c1 / (np.power(np.e, -beta * t) + c1)



# パラメータの設定
beta = 0.5 # 0.76282514433 # 9.76219788788
a = 0.0002
diff = 0.0001

# データの作成
maxT = 50
t = np.arange(0, maxT, diff)
y = func(a, beta, t) # for CDF
tPDF = np.arange(0, maxT - diff, diff)
yPDF = np.diff(y) / diff
print(yPDF)
# tPDF = tPDF - 4.75

# グラフの描画
# CDF
if (sys.argv[1] == "1"): 
    plt.plot(t, y, label="Data")
    plt.xlabel("Time")
    plt.ylabel("CDF")
    
    plt.legend()

    plt.xlim(xmin=0)
    plt.xlim(xmax=maxT + 1) 
    plt.ylim(ymin=0.0)
    plt.ylim(ymax=max(y) + 0.01)
    plt.show()
# PDF
else: 
    plt.plot(tPDF, yPDF, label="Data")
    plt.xlabel("Time")
    plt.ylabel("PDF")
    
    plt.legend()
    
    plt.xlim(xmin=0)
    plt.xlim(xmax=maxT + 1) 
    plt.ylim(ymin=0.0)
    plt.ylim(ymax=max(yPDF) + 0.01)
    plt.show()
