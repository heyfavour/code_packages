"""
原版是记忆池add新数据时就传入TD_ERROR存入，并且sample以后更新p 但是这样存在sample异常（新的数据无法被sample 并且一直sample loss大的数据）
莫凡的版本是新的数据都是最大。然后不断更新p 直到比 1小 然后越来越小概率被sample ,逻辑上更为合适一点
"""
import gym
import torch
import random
import numpy as np
import torch.nn as nn

# 定义参数
BATCH_SIZE = 128  # 每一批的训练量
LR = 0.001  # 学习率
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target的更新频率
MEMORY_CAPACITY = 1000  # memery较小时，训练较快，但是后期效果没记忆体大好，建议动态记忆体，前期记忆体小，后期记忆体大

env = gym.make('CartPole-v0')
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
N_ACTIONS = env.action_space.n  # 2 2个动作
N_STATES = env.observation_space.shape[0]  # 4 state的维度
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class SumTree:  # p越大，越容易sample到
    # idx tree的索引
    # write data的索引
    # idx = self.wirte + self.capacity - 1

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
    alpha = 0.6  # [0~1] Importance Sampling.alpha=0 退化为均匀采样
    epsilon = 0.01  # small amount to avoid zero priority

    beta = 0.4
    beta_increment_per_sampling = 0.001
    clip_error = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, sample):  # 新增的永远优先
        p = np.max(self.tree.tree[-self.tree.capacity:])
        if p == 0: p = self.clip_error  # 刚开始时没数据max_p=0
        self.tree.add(p, sample)

    def _get_priority(self, error):  # p的范围在[epsilon,clip_error]
        error = np.abs(error) + self.epsilon  # convert to abs and avoid 0
        clip_error = np.minimum(error, self.clip_error)
        return clip_error ** self.alpha  # error越大p越大 但是p有上线

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def sample(self, num):
        batch = []
        idxs = []
        priorities = []

        segment = self.tree.total() / num
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # 0.4 + 0.01->1

        for i in range(num):
            a = segment * i
            b = segment * (i + 1)
            sample = random.uniform(a, b)
            (idx, p, data) = self.tree.get(sample)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        # ISWeight = (N*Pj)^(-beta) / maxi_wi
        is_weight = np.power(self.tree.length * sampling_probabilities, -self.beta)  # (2000*p/p_sum)^-a MF （p/min_p)^-a
        is_weight = is_weight / is_weight.max()  #
        return batch, idxs, is_weight  #


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear": m.weight.data.normal_(0.0, 0.1)
    # m.bias.data.fill_(0)


class NET(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 20),
            nn.ReLU(),
            nn.Linear(20, N_ACTIONS),
        )
        self.apply(weights_init)  # initialization

    def forward(self, input):
        output = self.model(input)
        return output


class PER_DQN():
    def __init__(self):
        self.epsilon = 0.5
        self.epsilon_max = 0.99
        self.epsilon_step = 5000
        self.epsilon_decay = (self.epsilon_max - self.epsilon) / self.epsilon_step

        self.memory = Memory(MEMORY_CAPACITY)
        self.eval_net, self.target_net = NET(), NET()
        self.optimizer = torch.optim.AdamW(self.eval_net.parameters(), lr=LR)

        self.learn_step_counter = 0
        self.update_target_model()
        self.loss_func = nn.MSELoss()

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    # get action from model using epsilon-greedy policy
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:  # 贪婪策略
            action = self.predict_action(state)
        else:
            action = np.random.randint(0, N_ACTIONS)  # 随机产生一个action
        return action

    def predict_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            actions_value = self.eval_net(state)  # [1 2]
            action = actions_value.max(1)[1].item()
            return action

    # save sample (error,<s,a,r,s'>) to the replay memory
    def store_transition(self, state, action, reward, next_state, done):
        # transition = np.hstack((state, [action, reward], next_state))
        self.memory.add((state, action, reward, next_state, done))

    # pick samples from prioritized replay memory (with batch_size)
    def learn(self):
        # step 1 更新epsilon 慢慢提高策略的选取机率
        if self.epsilon < self.epsilon_max: self.epsilon = self.epsilon + self.epsilon_decay
        # step 2  每N步更新一次target_net
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0: self.update_target_model()
        self.learn_step_counter = self.learn_step_counter + 1

        batch, idxs, is_weight = self.memory.sample(BATCH_SIZE)
        batch = np.array(batch, dtype=object).transpose()

        sample_state = torch.FloatTensor(np.vstack(batch[0]))
        sample_action = torch.LongTensor(list(batch[1])).view(-1, 1)
        sample_reward = torch.FloatTensor(list(batch[2])).view(-1, 1)
        sample_next_state = torch.FloatTensor(np.vstack(batch[3]))
        sample_done = torch.FloatTensor(batch[4].astype(int)).view(-1, 1)

        q_eval = self.eval_net(sample_state).gather(1, sample_action)  # 去对应的acion的实际output

        q_next = self.target_net(sample_next_state).detach()
        # target = rewards + (1 - dones) * GAMMA * next_pred.max(1)[0]
        q_target = sample_reward + (1 - sample_done) * GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        error = torch.abs(q_eval - q_target).data.numpy()

        # update priority
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, error[i][0])

        self.optimizer.zero_grad()
        loss = (torch.FloatTensor(is_weight) * self.loss_func(q_eval, q_target)).mean()
        loss.backward()
        self.optimizer.step()


# modify the reward 如果不重定义分数，相当难收敛
def modify_reward(state):
    x, x_dot, theta, theta_dot = state  # (位置x，x加速度, 偏移角度theta, 角加速度)
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


if __name__ == "__main__":
    agent = PER_DQN()
    for epoch in range(400):
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
            if agent.memory.tree.length >= MEMORY_CAPACITY:
                agent.learn()
                if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
                # print(agent.memory.tree.tree[-MEMORY_CAPACITY:])
            if done:
                break
            state = next_state
