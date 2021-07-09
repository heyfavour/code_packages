"""
对数据生成的加速，不对最后的效果负责
"""
import random
import time
import threading
import numpy as np
import ray
import gym
import torch
import torch.nn as nn

from collections import deque

# from torch.cuda.amp import autocast,GradScaler #混合精度加速


FRAME_STACK = 4
#################### WORKER.PY ####################
BATCH_SIZE = 64
LEARNING_STARTS = 50000
SAVE_INTERVAL = 1000
TARGET_NETWORK_UPDATE_FREQ = 2500

GAMMA = 0.99
PRIORITIZED_REPLAY_ALPHA = 0.6
PRIORITIZED_REPLAY_BETA0 = 0.4
FORWARD_STEPS = 3  # N-STEP FORWARD

BUFFER_CAPACITY = 131072
MAX_EPISODE_LENGTH = 16384
SEQUENCE_LEN = 1024  # CUT ONE EPISODE TO SEQUENCES TO IMPROVE THE BUFFER SPACE UTILIZATION
#################### TRAIN.PY ####################
NUM_ACTORS = 16
BASE_EPS = 0.4
ALPHA = 0.7

ENV_NAME = 'CARTPOLE-V0'
env = gym.make(env_name)
env = env.unwrapped
N_ACTIONS = env.action_space.n  #
N_STATES = env.observation_space.shape[0]


########################################################################################POLICY
class Policy(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.pre_model = nn.Sequential(nn.Linear(N_STATES, 50))
        self.v_layers = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )
        self.q_layers = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS),
        )

    def forward(self, input):
        input = self.pre_model(input)
        v = self.v_layers(input)  # 环境的价值
        q = self.q_layers(input)  # 动作的价值
        # actions_value = v.expand_as(q) + (q - q.mean(dim=1,keepdim=True).expand_as(q)) 等同与下方
        actions_value = v + (q - q.mean(dim=1, keepdim=True))
        return actions_value


########################################################################################MEMERY
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

    def sum(self):
        return self.tree[0]


@ray.remote(num_cpus=1)
class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, buffer_capacity):
        self.tree = SumTree(buffer_capacity)

        self.alpha = 0.9  # [0~1] Importance Sampling.alpha=0 退化为均匀采样
        self.beta = 0.4
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.beta_increment_per_sampling = 0.001
        self.clip_error = 1.  # clipped abs error

        self.batched_data = []

        self.lock = threading.Lock()

    def add(self, sample):  # 新增的永远优先
        with self.lock:
            p = np.max(self.tree.tree[-self.tree.capacity:])
            if p == 0: p = self.clip_error  # 刚开始时没数据max_p=0
            self.tree.add(p, sample)

    def _get_priority(self, error):  # p的范围在[epsilon,clip_error]
        error = np.abs(error) + self.epsilon  # convert to abs and avoid 0
        clip_error = np.minimum(error, self.clip_error)
        return clip_error ** self.alpha  # error越大p越大 但是p有上线

    def update(self, idx, error):
        with self.lock:
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def sample(self, num):
        with self.lock:
            batch, idxs, priorities = [], [], []
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
            is_weight = np.power(self.tree.length * sampling_probabilities,
                                 -self.beta)  # (2000*p/p_sum)^-a MF （p/min_p)^-a
            is_weight = is_weight / is_weight.max()  #
            return batch, idxs, is_weight  #

    def run(self):
        self.background_thread = threading.Thread(target=self.sample_data, daemon=True)
        self.background_thread.start()

    def sample_data(self):
        while True:
            if len(self.batched_data) < 4:
                data = self.sample_batch(self.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)

    def get_data(self):
        if len(self.batched_data) == 0:
            data = self.sample_batch(self.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)


############################## Learner ##############################

@ray.remote(num_cpus=1)
class Agent:
    def __init__(self, memory):
        self.memory = memory
        self.eval_net, self.target_net = Policy(), Policy()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)

        self.learn_step_counter = 0
        self.update_target_model()
        self.loss_func = nn.MSELoss()

        self.store_weights()
        self.batch_size = 64

    def update_target_model(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

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

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        self.weights_id = ray.put(self.eval_net.state_dict())

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def learn(self):
        data_id = ray.get(self.buffer.get_data.remote())
        data = ray.get(data_id)
        if self.learn_step_counter % 5 == 0: self.update_target_model()
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
        q_target = sample_reward + (1 - sample_done) * 0.9 * q_next.max(1)[0].view(BATCH_SIZE, 1)
        error = torch.abs(q_eval - q_target).data.numpy()
        # update priority
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update.remote(idx, error[i][0])

        self.optimizer.zero_grad()
        loss = (torch.FloatTensor(is_weight) * self.loss_func(q_eval, q_target)).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

############################## Actor ##############################

class LocalBuffer:
    '''store transition of one episode'''

    def __init__(self, forward_steps=FORWARD_STEPS, frame_stack=FRAME_STACK, sequence_len=SEQUENCE_LEN, gamma=GAMMA):
        self.memory = np.zeros(forward_steps, dtype=object)
        self.memory_size = forward_steps
        self.memory_counter = 0

    def add(self, transition):  # data = (state, action, reward, next_state, done) 4 1 1 4 1
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def finish(self, last_q_val: np.ndarray = None):
        cumulated_gamma = [self.gamma ** self.forward_steps for _ in range(self.size - self.forward_steps)]
        # if last_q_val is None, means done
        if last_q_val:
            self.qval_buffer.append(last_q_val)
            cumulated_gamma.extend([self.gamma ** i for i in reversed(range(1, self.forward_steps + 1))])
        else:
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[-1]))
            cumulated_gamma.extend([0 for _ in range(self.forward_steps)])  # set gamma to 0 so don't need 'done'

        cumulated_gamma = np.array(cumulated_gamma, dtype=np.float16)
        self.obs_buffer = np.concatenate(self.obs_buffer)
        self.action_buffer = np.array(self.action_buffer, dtype=np.uint8)
        self.qval_buffer = np.concatenate(self.qval_buffer)
        self.reward_buffer = self.reward_buffer + [0 for _ in range(self.forward_steps - 1)]
        cumulated_reward = np.convolve(self.reward_buffer,
                                       [self.gamma ** (self.forward_steps - 1 - i) for i in range(self.forward_steps)],
                                       'valid').astype(np.float16)

        num_sequences = self.size // self.sequence_len + 1

        # td_errors
        td_errors = np.zeros(num_sequences * self.sequence_len, dtype=np.float32)
        max_qval = np.max(self.qval_buffer[self.forward_steps:self.size + 1], axis=1)
        max_qval = np.concatenate((max_qval, np.array([max_qval[-1] for _ in range(self.forward_steps - 1)])))
        target_qval = self.qval_buffer[np.arange(self.size), self.action_buffer]
        td_errors[:self.size] = np.abs(cumulated_reward + max_qval - target_qval).clip(1e-4)

        # cut one episode to sequences to improve the buffer space utilization
        sequences = []
        for i in range(0, num_sequences * self.sequence_len, self.sequence_len):
            obs = self.obs_buffer[i:i + self.sequence_len + 4]
            actions = self.action_buffer[i:i + self.sequence_len]
            rewards = cumulated_reward[i:i + self.sequence_len]
            td_error = td_errors[i:i + self.sequence_len]
            gamma = cumulated_gamma[i:i + self.sequence_len]

            sequences.append((obs, actions, rewards, gamma, td_error))

        return sequences


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, epsilon, agent, memory):
        self.env = gym.make(ENV_NAME)
        self.model = Policy()
        self.local_buffer = LocalBuffer()
        self.obs_history = deque([], maxlen=4)
        self.epsilon = epsilon
        self.agent = agent
        self.memory = memory

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:  # 贪婪策略
            action = self.predict_action(state)
        else:
            action = np.random.randint(0, N_ACTIONS)  # 随机产生一个action
        return action

    def predict_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            actions_value = self.model(state)
            action = actions_value.max(1)[1].item()
            return action

    def train(self):
        state = self.reset()
        while True:
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)

            self.local_buffer.add(action, reward, next_state, action)

            if done:
                sequences = self.local_buffer.finish()
                self.memory.add.remote(sequences)
                self.update_weights()#更新本地结构
                self.reset()

    def update_weights(self):
        weights_id = ray.get(self.agent.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)

    def reset(self):
        state = self.env.reset()
        self.local_buffer.reset()
        return state

if __name__ == '__main__':
    ray.init()
    memory = Memory.remote()
    agent = Agent.remote(memory)
    actors = [Actor.remote(0.95, agent, memory) for i in range(NUM_ACTORS)]

    for actor in actors: actor.train.remote()

    while not ray.get(memory.ready.remote()):
        time.sleep(5)
        ray.get(agent.stats.remote(5))

    print('start training')
    memory.run.remote()
    agent.run.remote()



