from __future__ import unicode_literals, division

import simpy
import numpy as np

class Car(object):

    # 時速が更新される単位時間
    step = 5

    def __init__(self, env, mean, std):
        # シミュレーション環境への参照
        self.env = env

        # 時速の平均値
        self.mean = mean
        # 時速の標準偏差
        self.std = std

        # 現在の時速
        self.velocity = 0.0
        # 現在位置
        self.location = 0.0
        # 直前の位置
        self.prev_location = 0.0

    def update_velocity(self):
        # 現在時速を更新
        # 現在時速は 平均 mean, 標準偏差 std の正規分布に従う
        v = np.random.normal(self.mean, self.std)
        if v < 0:
            v = 0
        return v

    def update_location(self):
        # 現在位置を更新
        self.prev_location = self.location
        self.location += self.velocity / 3600 * 1000
        return self.location

    def run(self):
        # 自身のプロセス (generator) を返すメソッド
        while True:
            if env.now % self.step == 0:
                # step が経過するたび、現在時速を更新 (乱数をふりなおす)
                self.velocity = self.update_velocity()
            form = '現在時刻 {0:2d} 位置: {1:.1f} m 時速 {2:.1f} km'
            print(form.format(self.env.now, self.location, self.velocity))
            self.update_location()
            yield self.env.timeout(1)

np.random.seed(1)

# 1. シミュレーション環境を作成
env = simpy.Environment()

# Car インスタンスを平均時速 72 km、標準偏差 10 km で作成
c = Car(env, mean=72, std=10)

# 2. シミュレーション対象のプロセス (generator) を作成
# 3. シミュレーション環境に プロセスを追加
env.process(c.run())

# 4. シミュレーション環境の実行 (100単位時間分)
env.run(until=100)
