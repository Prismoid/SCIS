from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

# 正規乱数から得たデータの分布が正規分布と差があるか検定
data = stats.norm.rvs(size=500)
pv = stats.kstest(data, "norm")[1]
print('p-value:' + str(pv))
# 正規分布の平均と分散を指定したい場合
mu = 5
sigma2 = 3
data = stats.norm.rvs(size=500) * sigma2 + mu
pv = stats.kstest(data, stats.norm(loc=mu, scale=sigma2).cdf)[1]
print('p-value:' + str(pv))

# 仮定する分布を変えてみる（平均値を1増やす）
# pv = stats.kstest(data, stats.expon(scale=600).cdf)[1]
# print('p-value:' + str(pv))

# 可視化
x = np.linspace(-5, 15, 100)
pdf_norm1 = stats.norm.pdf(x, loc=mu, scale=sigma2)
pdf_norm2 = stats.norm.pdf(x, loc=mu + 1, scale=sigma2)
plt.hist(data, density="norm")
plt.plot(x, pdf_norm1, 'r')
plt.plot(x, pdf_norm2, 'k')
plt.show()
