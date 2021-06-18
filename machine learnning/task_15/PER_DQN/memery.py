import random
import numpy as np
import numpy


class SumTree:  # p越大，越容易sample到
    # idx tree的索引
    # write data的索引
    # idx = self.wirte + self.capacity - 1

    wirte = 0  # 此时写入的位置

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
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
        self.data[self.wirte] = data
        idx = self.wirte + self.capacity - 1  # idx=>tree.idx
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


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    epsilon = 0.01  # small amount to avoid zero priority

    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha  # error越大p越大

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def sample(self, num):
        batch = []
        idxs = []
        segment = self.tree.total() / num
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # [1 0.4+0.001]

        for i in range(num):
            a = segment * i
            b = segment * (i + 1)
            sample = random.uniform(a, b)
            (idx, p, data) = self.tree.get(sample)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.length * sampling_probabilities, -self.beta)  # (2000*p/p_sum)^-0.4
        is_weight = is_weight / is_weight.max()  #
        return batch, idxs, is_weight  #


"""
class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
    
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p
    
    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
    
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
"""
if __name__ == '__main__':
    m = Memory(10)
    m.add(1, (1))
    for i in range(9):
        m.add(0.5, (0.5))
    m.sample(3)
