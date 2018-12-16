import numpy as np
import matplotlib.pyplot as plt

# 1. シミュレーション環境を作成

def findNonce(x):
    for i in range(1000000):
        y = np.random.rand()
        if (y < x):
            print("Hey")

findNonce(1 / 360000)
print(1 / 360000)


        


