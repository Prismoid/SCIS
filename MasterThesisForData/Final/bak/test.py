import matplotlib.pyplot as plt

lines = {'linestyle': 'None'}
plt.rc('lines', **lines)

plt.errorbar((0, 1), (1, 0), yerr=(0.1, 0.1), marker='o')

plt.savefig('test.pdf')
plt.delaxes()
plt.
