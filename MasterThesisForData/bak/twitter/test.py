import numpy as np
from pylab import *

data = []
#main
data.append(np.array([4006,2064,856,385,162,112,163,234,360,871,1657,2435,2903,2626,2790,2406,2366,2523,2841,2552,2803,2464,3421,4457], dtype='float'))
#sub
data.append(np.array([593,418,199,65,23,8,7,28,31,90,236,317,369,403,262,231,186,178,220,305,340,368,547,633], dtype='float'))
#friend 1
data.append(np.array([20,10,9,5,9,12,36,338,121,42,32,34,162,98,35,32,54,55,65,203,179,159,77,38], dtype='float'))
#friend 2
data.append(np.array([7458,3890,1293,323,235,280,673,2169,4569,5515,4976,4513,4225,4150,3416,3047,3179,3642,4296,4911,4278,5158,7282,7939], dtype='float'))
#friend 3
data.append(np.array([1254,803,307,176,48,61,231,512,528,554,750,704,702,549,575,533,560,549,456,553,759,922,1033,1317], dtype='float'))
#friend 4
data.append(np.array([816,382,161,23,8,23,61,227,460,621,523,603,820,730,596,580,569,619,596,770,792,857,950,937], dtype='float'))
#friend 5
data.append(np.array([2219,1429,912,479,256,189,246,370,624,691,921,878,1362,1053,879,950,1187,1039,1109,1221,1495,1580,2237,2236], dtype='float'))
#normalize
data_prob = [d / sum(d) for d in data]

xlim(0, 23)
plot(np.array(data_prob)[:7,:].T, linewidth=5)
legend('main sub friend1 friend2 friend3 friend4 friend5'.split())

show()
