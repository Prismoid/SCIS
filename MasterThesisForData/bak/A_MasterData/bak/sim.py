# http://sinhrks.hatenablog.com/entry/2014/12/14/005604

from __future__ import unicode_literals, division

import simpy
import numpy as np

# 1. シミュレーション環境を作成
env = simpy.Environment()

def car(env):
    # 時速 72 km = 秒速 20 m 
    velocity = 72 / 3600 * 1000
    # 現在位置
    location = 0.0
    
    # プロセス実行時に行われる処理
    while True:
        print('現在時刻 {0:2d} 位置: {1} m'.format(env.now, location))
        location += velocity
        # 次にプロセスを実行するまでのタイムアウト (ここでは毎回実行するため 1 を返す)
        yield env.timeout(1)

# 2. シミュレーション対象のプロセス (generator) を作成
car(env)
# <generator object car at 0x10c333640>

# 3. シミュレーション環境に プロセスを追加
env.process(car(env))

# 4. シミュレーション環境の実行 (15単位時間分)
env.run(until=15)

