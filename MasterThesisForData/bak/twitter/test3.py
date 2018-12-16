from scipy import stats
# stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='approx')
x = stats.norm.rvs(size=50)
y = stats.cauchy.rvs(size=50)
z = stats.t.rvs(3,size=50)
print(x)
print(stats.kstest(x,"norm").pvalue)
print(stats.kstest(y,"norm").pvalue)
print(stats.kstest(z,"norm").pvalue)
