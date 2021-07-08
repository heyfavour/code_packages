# -*- coding:utf8 -*-
import ray
import time, datetime
import asyncio

ray.init()

"""
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
"""
# ------------------------------------------------------------------------------------------
"""
# A regular Python function.
def my_function():
    return 1

# By adding the `@ray.remote` decorator, a regular Python function
# becomes a Ray remote function.
@ray.remote
def my_function():
    return 1

# To invoke this remote function, use the `remote` method.
# This will immediately return an object ref (a future) and then create
# a task that will be executed on a worker process.
obj_ref = my_function.remote()

# The result can be retrieved with ray.get.
assert ray.get(obj_ref) == 1


@ray.remote
def slow_function():
    time.sleep(3)
    return 1

# Invocations of Ray remote functions happen in parallel.
# All computation is performed in the background, driven by Ray's internal event loop.
print(datetime.datetime.now())
d = [slow_function.remote() for _ in range(4)]
print(datetime.datetime.now())
ray.get(d)
print(datetime.datetime.now())
"""
# ------------------------------------------------------------------------------------------指定资源
"""
ray.init(num_cpus=8, num_gpus=4, resources={'Custom': 2})#指定资源

# Specify required resources.
@ray.remote(num_cpus=4, num_gpus=2)
def my_function():
    return 1

# Ray also supports fractional resource requirements.
@ray.remote(num_gpus=0.5)
def h():
    return 1

# Ray support custom resources too.
@ray.remote(resources={'Custom': 1})
def f():
    return 1
"""
# ------------------------------------------------------------------------------------------多参数返回
"""
@ray.remote(num_returns=3)
def return_multiple():
    return 1, 2, 3

a, b, c = return_multiple.remote()
"""
# ------------------------------------------------------------------------------------------多参数返回
"""
@ray.remote
def blocking_operation():
    time.sleep(100)
"""
# ------------------------------------------------------------------------------------------取消任务
"""
obj_ref = blocking_operation.remote()
ray.cancel(obj_ref)

from ray.exceptions import TaskCancelledError

try:
    ray.get(obj_ref)
except TaskCancelledError:
    print("Object reference was cancelled.")
"""
# ------------------------------------------------------------------------------------------共享内存
"""
obj_ref = ray.put(1)
print(ray.get(obj_ref))
# Get the values of multiple object refs in parallel.
print(ray.get([ray.put(i) for i in range(3)]))
"""
# ------------------------------------------------------------------------------------------超时
"""
# You can also set a timeout to return early from a ``get`` that's blocking for too long.
from ray.exceptions import GetTimeoutError

@ray.remote
def long_running_function():
    time.sleep(8)

obj_ref = long_running_function.remote()
try:
    ray.get(obj_ref, timeout=4)
except GetTimeoutError:
    print("`get` timed out.")
"""
# ------------------------------------------------------------------------------------------等待完成
# ready_refs, remaining_refs = ray.wait(object_refs, num_returns=1, timeout=None)
# ------------------------------------------------------------------------------------------
"""
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

Counter = ray.remote(Counter)  #Actor类

counter_actor = Counter.remote()#实例化

counter_actor = Counter.remote()

assert ray.get(counter_actor.increment.remote()) == 1

@ray.remote
class Foo(object):

    # Any method of the actor can return multiple object refs.
    @ray.method(num_returns=2)
    def bar(self):
        return 1, 2

f = Foo.remote()

obj_ref1, obj_ref2 = f.bar.remote()
assert ray.get(obj_ref1) == 1
assert ray.get(obj_ref2) == 2
"""
# ------------------------------------------------------------------------------------------资源
"""
ray.get_gpu_ids()
@ray.remote(num_cpus=2, num_gpus=1)
class GPUActor(object):
    pass

@ray.remote(num_cpus=4)
class Counter(object):
    ...
#不同资源需求得acotr
a1 = Counter.options(num_cpus=1, resources={"Custom1": 1}).remote()
a2 = Counter.options(num_cpus=2, resources={"Custom2": 1}).remote()
a3 = Counter.options(num_cpus=3, resources={"Custom3": 1}).remote()
"""
# ------------------------------------------------------------------------------------------传递actor
"""
@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

@ray.remote
def f(counter):
    for _ in range(1000):
        time.sleep(0.1)
        counter.increment.remote()

counter = Counter.remote()

# Start some tasks that use the actor.
[f.remote(counter) for _ in range(3)]#这里永远自增得是counter得value
while True:
    print(ray.get(counter.get_counter.remote()))
"""
# ------------------------------------------------------------------------------------------NAMESPACE
"""
# Create an actor with a name
counter = Counter.options(name="some_name").remote()
#通过命名空间获取acotr
# Retrieve the actor later somewhere
counter = ray.get_actor("some_name")


import ray

@ray.remote
class Actor:
  pass

# driver_1.py
ray.init(address="auto", namespace="colors")
Actor.options(name="orange", lifetime="detached")

# driver_2.py 连向另一个命名空间
ray.init(address="auto", namespace="fruit")
# orange 在colors中所以无法获取
ray.get_actor("orange")

# driver_3.py
#连向colors,可以获取orange
ray.init(address="auto", namespace="colors")



----------------------------------------------------------
ray.init(namespace="hello")
# or using ray client
ray.client().namespace("world").connect()

import ray
ray.init(address="auto", namespace="colors")
# Will print the information about "colors" namespace
print(ray.get_runtime_context().namespace)
"""
# ------------------------------------------------------------------------------------------生命周期
"""
counter = Counter.options(name="CounterActor", lifetime="detached").remote()

#上方代码退出后更然可以获取
counter = ray.get_actor("CounterActor")
print(ray.get(counter.get_counter.remote()))
"""
# ------------------------------------------------------------------------------------------POOL
"""
from ray.util import ActorPool

a1, a2 = Actor.remote(), Actor.remote()
pool = ActorPool([a1, a2])
print(pool.map(
                lambda a, v: a.double.remote(v), [1, 2, 3, 4]
              )
     )
# [2, 4, 6, 8]
"""
# ------------------------------------------------------------------------------------------AsyncIO
"""
@ray.remote
class AsyncActor:
    # multiple invocation of this method can be running in
    # the event loop at the same time
    async def run_concurrent(self):
        print("started")
        await asyncio.sleep(2) # concurrent workload here
        print("finished")

actor = AsyncActor.remote()
# regular ray.get
#ray.get([actor.run_concurrent.remote() for _ in range(4)])
# async ray.get
#a = await actor.run_concurrent.remote()
#print(a)
"""
# ------------------------------------------------------------------------------------------GPU
"""
import os

@ray.remote(num_cpus=0.25)
def use_gpu():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
ray.get([use_gpu.remote() for _ in range(4)])
"""
# ------------------------------------------------------------------------------------------序列化
# ------------------------------------------------------------------------------------------LOCK
"""
from filelock import FileLock

@ray.remote
def write_to_file(text):
    # Create a filelock object. Consider using an absolute path for the lock.
    with FileLock("my_data.txt.lock"):
        with open("my_data.txt","a") as f:
            f.write(text)


ray.get([write_to_file.remote("hi there!\n") for i in range(3)])

with open("my_data.txt") as f:
    print(f.read())
"""
# ------------------------------------------------------------------------------------------消息传递
"""
# Also available via `from ray.test_utils import SignalActor`
import ray
import asyncio

@ray.remote(num_cpus=0)
class SignalActor:
    def __init__(self):
        self.ready_event = asyncio.Event()

    def send(self, clear=False):
        self.ready_event.set()
        if clear:
            self.ready_event.clear()

    async def wait(self, should_wait=True):
        if should_wait:
            await self.ready_event.wait()

@ray.remote
def wait_and_go(signal):
    ray.get(signal.wait.remote())

    print("go!")

ray.init()
signal = SignalActor.remote()
tasks = [wait_and_go.remote(signal) for _ in range(4)]
print("ready...")
# Tasks will all be waiting for the signals.
print("set..")
ray.get(signal.send.remote())

# Tasks are unblocked.
ray.get(tasks)

##  Output is:
# ready...
# get set..

# (pid=77366) go!
# (pid=77372) go!
# (pid=77367) go!
# (pid=77358) go!
from ray.util.queue import Queue

# You can pass this object around to different tasks/actors
queue = Queue(maxsize=100)

@ray.remote
def consumer(queue):
    next_item = queue.get(block=True)
    print(f"got work {next_item}")


[queue.put(i) for i in range(10)]
consumers = [consumer.remote(queue) for _ in range(2)]
"""
# ------------------------------------------------------------------------------------------pytorch demo
# https://docs.ray.io/en/master/using-ray-with-pytorch.html
# ------------------------------------------------------------------------------------------关闭
# ray.actor.exit_actor()
# ray.kill(actor_handle)
ray.shutdown()
