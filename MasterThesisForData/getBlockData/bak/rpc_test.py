# coding:utf-8
from web3 import Web3, HTTPProvider, IPCProvider

web3 = Web3(HTTPProvider('http://localhost:8545'))
# web3 = Web3(IPCProvider())
blockNumber = web3.eth.blockNumber
main_user = web3.eth.accounts[0]
testSha3 = web3.sha3("0x74657484")
# testSha3 = web3.soliditySha3('bool', True)
# testSha3 = Web3.soliditySha3(['bool'], [True])
# (['address'], ["0x49eddd3769c0712032808d86597b84ac5c2f5614"])
print(blockNumber)
print(main_user)
print(testSha3)
# psn.unlockAccount(obj.eth.accounts[0],"123456",1000)

## https://gist.github.com/bas-vk/d46d83da2b2b4721efb0907aecdb7ebdに基づく ##

def messageHash(msg):
    return web3.sha3('\x19Ethereum Signed Message:\n' + msg.length + msg)

accountToSignWith = web3.eth.accounts[0]
message = b"Hello"

# コントラクトへのアクセス定義
# contractABI = [{"constant":true,"inputs":[{"name":"msgHash","type":"bytes32"},{"name":"v","type":"uint8"},{"name":"r","type":"bytes32"},{"name":"s","type":"bytes32"}],"name":"RecoverAddress","outputs":[{"name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"}]
contractAddress = "0xc4eba9b36793e01002a350742b7d16a6877c1019"
# signAndVerifyContract = web3.eth.contract(contractABI).at(contractAddress)

# 署名の実行
key = "\xb2\\}\xb3\x1f\xee\xd9\x12''\xbf\t9\xdcv\x9a\x96VK-\xe4\xc4rm\x03[6\xec\xf1\xe5\xb3d"
web3.eth.account.sign(0x49e299a55346, key)
