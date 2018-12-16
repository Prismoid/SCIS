import numpy as np
import matplotlib.pyplot as plt
import math

x = np.loadtxt('z001newData4200000.dat')
xN = [0] * 5000
for i in range(5000):
    xN[i] = x[(i + 1) * 100] - x[i * 100]

start = 0
end = 120

plt.xlabel("Time since last block [s]")
plt.ylabel("PDF")
plt.title("Block vs. time")
plt.xlim([0, 3000])
plt.ylim([0, 1.08])

plt.plot
plt.hist(xN, bins=50,normed=True, color="#000080", ec='black')
plt.hist(xN, bins=1000,normed=True, histtype='step', cumulative=True, label='CDF')
plt.show()
