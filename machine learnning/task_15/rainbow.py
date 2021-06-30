"""
RAINBOW 7种混合
1.DDQN
2.duelingDQN
3.noisy dqn 参数越来越小 但是也能保持较好的结果
4.categorical dqn
5.N-step
6.PER 网络上都没有看到让人很好理解得计算p和categorical dqn相结合得案例，所以这部分是自己写的
"""
import time

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
        return (self.buffer[0][0], self.buffer[0][1], sum_rewards, next_state, done)

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
        is_weight = torch.FloatTensor(is_weight / is_weight.max()).view(-1, 1)
        return batch, idxs, is_weight  # ISWeight = (N*Pj)^(-beta) / maxi_wi

    def _get_priority(self, error):  # p的范围在[epsilon,clip_error]
        error = np.abs(error) + self.epsilon  # convert to abs and avoid 0
        clip_error = np.minimum(error, self.clip_error)
        return clip_error ** self.alpha  # error越大p越大 但是p有上线

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def update_batch(self, idxs, error):
        for i, id in enumerate(idxs):
            self.update(id, error[i])

    def __len__(self):
        return self.tree.length


class Memory(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=MEMORY_CAPACITY, n_step=2):
        self.memory = np.zeros(memory_size, dtype=object)
        self.memory_size = memory_size
        self.memory_counter = 0
        self.n_step = n_step

    def add(self, transition):  # data = (state, action, reward, next_state, done) 4 1 1 4 1
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def sample(self, batch_size=BATCH_SIZE):
        sample_index = np.random.choice(self.memory_size, batch_size)  # 2000中取一个batch index [batch_size]
        # [(s a r s) ] -> [(s) (a) (r) (s)]
        sample_memory = np.array(list(self.memory[sample_index]), dtype=object).transpose()
        sample_state = torch.FloatTensor(np.vstack(sample_memory[0]))
        sample_action = torch.LongTensor(list(sample_memory[1])).view(-1, 1)
        sample_reward = torch.FloatTensor(list(sample_memory[2])).view(-1, 1)
        sample_next_state = torch.FloatTensor(np.vstack(sample_memory[3]))
        sample_done = torch.FloatTensor(sample_memory[4].astype(int)).view(-1, 1)
        return sample_state, sample_action, sample_reward, sample_next_state, sample_done

    def __len__(self):
        return min(self.memory_counter, MEMORY_CAPACITY)


from noisy_DQN import NoisyLinear  # noisy_dqn


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.v_noisy_layer_mid = NoisyLinear(128, 128)
        self.v_noisy_layer_out = NoisyLinear(128, ATOMS_NUM)

        self.q_noisy_layer_mid = NoisyLinear(128, 128)
        self.q_noisy_layer_out = NoisyLinear(128, N_ACTIONS * ATOMS_NUM)

        self.pre_model = nn.Sequential(
            nn.Linear(N_STATES, 128),
            nn.ReLU(),
        )
        self.apply(weights_init)  # initialization

    def forward(self, input):
        distribution = self.dist(input) * SUPPORT
        output = torch.sum(distribution, dim=2)
        return output

    def dist(self, input):
        mid = self.pre_model(input)
        v = self.v_noisy_layer_out(F.relu(self.v_noisy_layer_mid(mid))).view(-1, 1, ATOMS_NUM)  # 环境的价值
        q = self.q_noisy_layer_out(F.relu(self.q_noisy_layer_mid(mid))).view(-1, N_ACTIONS, ATOMS_NUM)  # 动作的价值
        output = v + (q - q.mean(dim=1, keepdim=True))  # [BATCH N_ACTIONS ATOMS_NUM]
        distribution = F.softmax(output, dim=-1)
        return distribution

    def reset_noise(self):
        self.v_noisy_layer_mid.reset_noise()
        self.v_noisy_layer_out.reset_noise()
        self.q_noisy_layer_mid.reset_noise()
        self.q_noisy_layer_out.reset_noise()


class Agent():
    def __init__(self):
        self.memory = PERMemery(MEMORY_CAPACITY, 2)
        self.eval_net, self.target_net = Net(), Net()
        self.optimizer = torch.optim.AdamW(self.eval_net.parameters(), lr=LR)

        self.learn_step_counter = 0
        self.update_target_model()

    def update_target_model(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state):  # 使用noisy以后总是贪婪的使用noisy选取的动作 不再需要epsilon
        action = self.predict_action(state)
        return action

    def predict_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            actions_value = self.eval_net(state)  # [1 2]
            action = actions_value.max(1)[1].item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def cal_dist(self, next_state, sample_reward, sample_done):
        delta_z = (V_MAX - V_MIN) / (ATOMS_NUM - 1)
        next_dist = self.target_net.dist(
            next_state).detach() * SUPPORT  # [32 2 51] = [BATCH N_ACTION softmax(N_ATOMS)]*[分布]
        next_action = next_dist.sum(2).max(1)[1].view(-1, 1, 1).expand(BATCH_SIZE, 1, ATOMS_NUM)  # [32]=>[32 1 51]
        next_dist = next_dist.gather(1, next_action).view(BATCH_SIZE, -1)  # [32 N_ATOMS] 根据action 获取执行action的分布
        sample_reward = sample_reward.expand_as(next_dist)  # [BATCH N_ATOMS] [BATCH R.. R] ATOMS_NUM个R
        sample_done = sample_done.expand_as(next_dist)
        support = SUPPORT.unsqueeze(0).expand_as(next_dist)  # [BATCH N_ATOMS]
        Tz = (sample_reward + (1 - sample_done) * (GAMMA ** self.memory.n_step) * support).clamp(min=V_MIN, max=V_MAX)
        b = (Tz - V_MIN) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (BATCH_SIZE - 1) * ATOMS_NUM, BATCH_SIZE).long().unsqueeze(1).expand(BATCH_SIZE,
                                                                                                        ATOMS_NUM)

        dist = torch.zeros(next_dist.size())
        dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        return dist

    def _compute_loss(self, batch):
        batch = np.array(batch, dtype=object).transpose()
        sample_state = torch.FloatTensor(np.vstack(batch[0]))
        sample_action = torch.LongTensor(list(batch[1])).view(-1, 1)
        sample_reward = torch.FloatTensor(list(batch[2])).view(-1, 1)
        sample_next_state = torch.FloatTensor(np.vstack(batch[3]))
        sample_done = torch.FloatTensor(batch[4].astype(int)).view(-1, 1)
        target_dist = self.cal_dist(sample_next_state, sample_reward, sample_done)

        eval_action = sample_action.view(-1, 1, 1).expand(-1, 1, ATOMS_NUM)  # [BATCH 1 ATOMS_NUM]
        eval_dist = self.eval_net.dist(sample_state).gather(1, eval_action).squeeze(1)

        eval_dist.data.clamp_(0.01, 0.99)
        loss = -(target_dist * eval_dist.log()).sum(1)
        error = torch.mean(abs(eval_dist - target_dist),dim=1).data.numpy()
        # error = -(target_dist * eval_dist.log()).sum(1).mean()
        return loss, error  # Wasserstein  lnp*Reward

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0: self.update_target_model()
        self.learn_step_counter = self.learn_step_counter + 1

        batch, idxs, is_weight = self.memory.sample(BATCH_SIZE)  # PER
        loss, error = self._compute_loss(batch)
        loss = torch.mean(is_weight * loss)
        agent.loss.append(loss.item())
        self.memory.update_batch(idxs, error)  # PER: update priority
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 10)  # 防止梯度爆炸
        self.optimizer.step()

    def reset_noise(self, ):
        self.eval_net.reset_noise()
        self.update_target_model()


def modify_reward(state):
    x, x_dot, theta, theta_dot = state  # (位置x，x加速度, 偏移角度theta, 角加速度)
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


def plot(rewards, loss):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('rewards')

    plt.plot(rewards)

    plt.title('loss')
    plt.subplot(132)
    plt.plot(loss)
    plt.show()


if __name__ == '__main__':
    agent = Agent()
    agent.epoch_rewards = []
    agent.loss = []
    for epoch in range(20000):
        state = env.reset()
        epoch_rewards = 0
        while True:
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            reward = modify_reward(next_state)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            epoch_rewards = epoch_rewards + reward
            if len(agent.memory) >= MEMORY_CAPACITY:
                agent.learn()
                if done:
                    print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}| Tree Mean {agent.memory.tree.tree[-MEMORY_CAPACITY:].mean()}')
                    print(agent.memory.tree.tree[-MEMORY_CAPACITY:].max())
                    print(agent.memory.tree.tree[-MEMORY_CAPACITY:])
            if done:
                break
            state = next_state
        agent.reset_noise()
        agent.epoch_rewards.append(epoch_rewards)

        if epoch % 5 == 0 and len(agent.memory) >= MEMORY_CAPACITY:plot(agent.epoch_rewards, agent.loss)
