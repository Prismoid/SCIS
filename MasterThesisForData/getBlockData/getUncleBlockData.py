# coding:utf-8
from web3 import Web3, HTTPProvider, IPCProvider

web3 = Web3(HTTPProvider('http://localhost:8545'))

# f1 = open('uncleData4600000.txt', 'a')
# f2 = open('tsData4600000.txt', 'a')
n = 4600000

uncleSum = 0
uncleDaySum = 0
for i in range(1001): 
    block = web3.eth.getBlock(n + i)
    ts = block.timestamp                                                                                                                                                            
    if len(block.uncles) > 0:
        print(block.uncles[0])
        print(block)
        break


# f1.close()
# f2.close()

