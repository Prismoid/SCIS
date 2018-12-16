import numpy as np
import matplotlib.pyplot as plt
import math

def expDist(lamb, time):
    return 1 - np.exp(-lamb * time)


def funcTime(lamb, prob):
    timeForCalc = 100
    incr = 100
    now = 0
    flag = 0
    while (np.abs(now - prob) > 0.001 * 0.0001):
        now = expDist(lamb, timeForCalc)
        timeForCalc += incr
        if (flag == 0 and now > prob): # 0.005                                                                                                                                       
            flag = 1
            incr = - 1 * incr / 2
        elif (flag == 1 and now < prob):
            flag = 0
            incr = - 1 * incr / 2
    return timeForCalc

def func(t, lamb):
    return lamb * np.power(math.exp(1), ( - lamb * t))

lamb1 = 1 / 15
lamb2 = 1 / 20
start = 0
end = 200
stride = 0.2
T = np.arange(start, end, stride)
Out1 = func(T, lamb1)
Out2 = func(T, lamb2)


plt.xlabel("Time since last block [s]",fontsize=25)
plt.ylabel("PDF",fontsize=25)
plt.title("Block vs. time",fontsize=25)
plt.xlim([start, 100])
plt.ylim([0, 0.08])
plt.tick_params(labelsize = 25)

x150 = funcTime(lamb1, 0.5)
x190 = funcTime(lamb1, 0.9)
x250 = funcTime(lamb2, 0.5)
x290 = funcTime(lamb2, 0.9)
y = [0.0, 0.06]
print(x150)
print(x190)

plt.plot(T, Out1, color="#00ff00", label=r"$\lambda = 1/15$")
plt.plot([x150, x150], y, color="#00ff00", linestyle="dashed")
plt.plot([x190, x190], y, color="#00ff00", linestyle="dashed")
plt.plot(T, Out2, color="#0000ff", label=r"$\lambda = 1/20$")
# plt.plot([x250, x250], y, color="#0000ff", dasth)
# plt.plot([x290, x290], y, color="#0000ff")
plt.legend(fontsize = 25)
plt.show()
