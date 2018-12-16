import numpy as np
import matplotlib.pyplot as plt
import math

x = np.loadtxt('z000tsData4100000-4600000.dat')
diff = np.diff(x)
maxV = 0
print(diff)
for i in range(500000): 
    if diff[i] > maxV:
        maxV = diff[i]

def func(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

num = 1000
lamb = 1 / ((x[500000] - x[0]) / 500000)
start = 0
end = 200
stride = 0.2
num = (end - start) / stride + 1
T = np.arange(start, end, stride)
Out = func(T, lamb)

print(maxV)
print(1/lamb)
plt.xlabel("Time since last block [s]",fontsize=20)
plt.ylabel("PDF",fontsize=20)
plt.title("Block vs. time",fontsize=20)
plt.xlim([start, 120])
plt.ylim([0, 0.08])
plt.tick_params(labelsize = 15)


plt.plot(T, Out, color="#ff0000")
plt.plot
plt.hist(diff - 1, bins=50,normed=True, color="#000080", ec='black')
plt.show()
