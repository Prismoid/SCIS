import numpy as np
import matplotlib.pyplot as plt
import math

x1 = np.loadtxt('./bakGetBlockData/tsData4600000-4700000.txt')
x2 = np.loadtxt('./bakGetBlockData/tsData4700000-5100000.txt')
f = open('tsData4600000-5100000.dat', 'a')
for i in range(100000): 
    f.write(str(x1[i]) + " ")
for i in range(400001):
    f.write(str(x2[i]) + " ")

        
'''
def func(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

num = 1000
lamb = 1 / ((x[500000] - x[0]) / 500000)
start = 0
end = 120
stride = 0.2
num = (end - start) / stride + 1
T = np.arange(start, end, stride)
Out = func(T, lamb)


print(1/lamb)
plt.xlabel("Time since last block [s]")
plt.ylabel("PDF")
plt.title("Block vs. time")
plt.xlim([0, 120])
plt.ylim([0, 0.08])

plt.plot(T, Out, color="#ff0000")
plt.plot
plt.hist(diff - 1, bins=50,normed=True, color="#000080", ec='black')
plt.show()
'''
