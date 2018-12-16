# coding:utf-8
from web3 import Web3, HTTPProvider, IPCProvider

web3 = Web3(HTTPProvider('http://localhost:8545'))
# web3 = Web3(IPCProvider())

f = open('uncleData.txt', 'a')
f = open('uncleDayData.txt', 'a')
n = 4200000

uncleSum = 0
uncleDaySum = 0
for i in range(15): 
    block = web3.eth.getBlock(n + i)
#    if block.uncles != []: 
        # print(block.uncles)
    if i % 10 == 0:
        f.write(str(uncleDaySum) + ' ')
        uncleDaySum = 0
    if len(block.uncles) > 0:
#        print(len(block.uncles))
        uncleSum += len(block.uncles)
        uncleDaySum += len(block.uncles)

f.write(str(uncleSum) + '')

#for i in range(7200):
#    block = web3.eth.getBlock(n + i)
#    ts = block.timestamp
#    f.write(str(ts) + ' ')
#    if (i % 1000) == 0:
#        print(str(i) + "\n")

# f.close()

