# -*- coding:utf8 -*-
import ray

ray.init()


@ray.remote
def f(x):
    return x * x


futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))  # [0, 1, 4, 9]


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n = self.n + 1

    def read(self):
        return self.n


counters = [Counter.remote() for i in range(4)]
[c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]#获取objectRef get 获取object
print(futures)
print(ray.get(futures))  # [1, 1, 1, 1]
