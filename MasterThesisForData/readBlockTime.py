import numpy as np
import matplotlib.pyplot as plt
import math

x = np.loadtxt('./Final/z001tsData4600000-5100000.dat')
diff = np.diff(x)
maxV = 0
print(diff)
for i in range(500000): 
    if diff[i] > maxV:
        maxV = diff[i]

def func(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

num = 1000
lamb  = 1 / ((x[500000] - x[0]) / 500000)
lamb1 = lamb * 1.058
lamb2 = lamb * 0.942
start = 0
end = 200
stride = 0.2
num = (end - start) / stride + 1
T = np.arange(start, end, stride)
Out  = func(T, lamb)
Out1 = func(T, lamb1)
Out2 = func(T, lamb2)

print(maxV)
print(1/lamb)
plt.xlabel("Time since last block [s]")
plt.ylabel("PDF")
plt.title("Block vs. time")
plt.xlim([start, 120])
plt.ylim([0, 0.08])

plt.plot(T, Out, color="#ff0000")
plt.plot(T, Out1, color="#00ff00")
plt.plot(T, Out2, color="#0000ff")
plt.plot
plt.hist(diff - 1, bins=50,normed=True, color="#000080", ec='black')
plt.show()
