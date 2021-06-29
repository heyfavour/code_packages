"""
RAINBOW 7种混合
1.DDQN
2.duelingDQN
3.noisy dqn 参数越来越小 但是也能保持较好的结果
4.categorical dqn
5.N-step
6.PER
"""
import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

BATCH_SIZE = 128  # 每一批的训练量
LR = 0.001  # 学习率
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target的更新频率
MEMORY_CAPACITY = 2000  # memery较小时，训练较快，但是后期效果没记忆体大好，建议动态记忆体，前期记忆体小，后期记忆体大

env = gym.make('CartPole-v0')
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
N_ACTIONS = env.action_space.n  # 2 2个动作
N_STATES = env.observation_space.shape[0]  # 4 state的维度
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

ATOMS_NUM = 51
V_MIN = -10
V_MAX = 10
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOMS_NUM)


class SumTree:  # p越大，越容易sample到
    wirte = 0  # 此时写入的位置

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.length = 0

    def _propagate(self, idx, change):  # 递归将变化更新到parent_node
        parent = (idx - 1) // 2
        self.tree[parent] = self.tree[parent] + change  # 将变化写入父节点
        if parent != 0: self._propagate(parent, change)  # p值更新后，其上面的父节点也都要更新

    def update(self, idx, p):
        change = p - self.tree[idx]  # 当前p-原来存的p
        self.tree[idx] = p  # 节点写入新的p
        self._propagate(idx, change)

    def add(self, p, data):  # p sample的概率，也是error的大小 data state 需要储存的数据
        idx = self.wirte + self.capacity - 1  # idx=>tree.idx

        self.data[self.wirte] = data
        self.update(idx, p)

        self.wirte = self.wirte + 1
        if self.wirte >= self.capacity: self.wirte = 0  # 叶结点存满了就从头开始覆写
        if self.length < self.capacity: self.length = self.length + 1

    # find sample on leaf node
    def _retrieve(self, idx, sample):  ## 检索s值
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): return idx  ## 说明已经超过叶结点了 返回传入的idx len(tree)=2 * capacity - 1
        if sample <= self.tree[left]:  # 比左边小 传到左
            return self._retrieve(left, sample)  # 递归调用
        else:  # 比左边大，sample-p
            return self._retrieve(right, sample - self.tree[left])

    def get(self, sample):
        idx = self._retrieve(0, sample)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])  # idx p (s a r s)

    def total(self):
        return self.tree[0]


class PERMemery():
    alpha = 0.9  # [0~1] Importance Sampling.alpha=0 退化为均匀采样
    epsilon = 0.01  # small amount to avoid zero priority
    beta = 0.4
    beta_increment_per_sampling = 0.001
    clip_error = 1.  # clipped abs error

    def __init__(self, capacity=MEMORY_CAPACITY, n_step=2):
        self.tree = SumTree(capacity)
        self.buffer = deque(maxlen=n_step)
        self.n_step = n_step

    def get_n_step(self):  # [s a r s a]
        sum_rewards = 0
        for i in range(self.n_step):  # 其他进入的
            (action, state, reward, next_state, done) = self.buffer[i]
            sum_rewards = sum_rewards + (GAMMA ** i) * reward * (1 - done)
            if done: break
        return (self.buffer[0], self.buffer[1], sum_rewards, next_state, done)

    def add(self, transition):  # 新增的永远优先
        self.buffer.append(transition)
        if len(self.buffer) < self.n_step: return
        transition = self.get_n_step()
        p = np.max(self.tree.tree[-self.tree.capacity:])
        if p == 0: p = self.clip_error  # 刚开始时没数据max_p=0
        self.tree.add(p, transition)

    def sample(self, num):
        batch, idxs, priorities, segment = [], [], [], self.tree.total() / num
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # 0.4 + 0.01->1
        for i in range(num):
            sample = random.uniform(segment * i, segment * (i + 1))
            (idx, p, data) = self.tree.get(sample)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.length * sampling_probabilities, -self.beta)  # (2000*p/p_sum)^-a MF （p/min_p)^-a
        is_weight = is_weight / is_weight.max()  #
        return batch, idxs, is_weight  # ISWeight = (N*Pj)^(-beta) / maxi_wi


if __name__ == '__main__':
    m = PERMemery()
