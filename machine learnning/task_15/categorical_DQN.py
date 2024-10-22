"""
对累积回报这个随机变量的分布 Z(x,a) 进行建模，而非只建模其期望。
"""
import torch
import torch.nn as nn
import numpy as np
import gym

import torch.nn.functional as F

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 1000
# env = gym.make('Pendulum-v0')
env = gym.make('CartPole-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.n  #
N_STATES = env.observation_space.shape[0]

ATOMS_NUM = 51
V_MIN = -10
V_MAX = 10
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOMS_NUM)


class Memory(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=MEMORY_CAPACITY):
        self.memory = np.zeros(memory_size, dtype=object)
        self.memory_size = memory_size
        self.memory_counter = 0

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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, N_ACTIONS * ATOMS_NUM),
        )

    def forward(self, input):
        distribution = self.dist(input) * SUPPORT
        output = torch.sum(distribution, dim=2)
        return output

    def dist(self, input):  # => [batch action_dim softmax(ATOMS_NUM)]*[分布]
        distribution = F.softmax(self.model(input).view(-1, N_ACTIONS, ATOMS_NUM), dim=-1)
        return distribution


class Categorical_DQN():
    def __init__(self):
        self.epsilon = EPSILON

        self.memory = Memory(MEMORY_CAPACITY)
        self.eval_net, self.target_net = Net(), Net()
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
            actions = self.eval_net(state).detach()  # [1 2 51]
            action = actions.max(1)[1].item()
            return action

    # save sample (error,<s,a,r,s'>) to the replay memory
    def store_transition(self, state, action, reward, next_state, done):
        # transition = np.hstack((state, [action, reward], next_state))
        self.memory.add((state, action, reward, next_state, done))

    def cal_dist(self, next_state, sample_reward, sample_done):
        delta_z = (V_MAX - V_MIN) / (ATOMS_NUM - 1)
        support = torch.linspace(V_MIN, V_MAX, ATOMS_NUM)

        next_dist = self.target_net.dist(next_state).detach()*SUPPORT  # [32 2 51] = [BATCH N_ACTION softmax(N_ATOMS)]*[分布]
        next_action = next_dist.sum(2).max(1)[1].view(-1, 1, 1).expand(BATCH_SIZE, 1, ATOMS_NUM)  # [32]=>[32 1 51]
        next_dist = next_dist.gather(1, next_action).view(BATCH_SIZE, -1)  # [32 N_ATOMS] 根据action 获取执行action的分布

        sample_reward = sample_reward.expand_as(next_dist)  # [BATCH N_ATOMS] [BATCH R.. R] ATOMS_NUM个R
        sample_done = sample_done.expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)  # [BATCH N_ATOMS]

        Tz = (sample_reward + (1 - sample_done) * GAMMA * support).clamp(V_MIN, max=V_MAX)
        b = (Tz - V_MIN) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (BATCH_SIZE - 1) * ATOMS_NUM, BATCH_SIZE
                                ).long().unsqueeze(1).expand(BATCH_SIZE,ATOMS_NUM)

        dist = torch.zeros(next_dist.size())
        dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        return dist

    def learn(self):
        # step 1  每N步更新一次target_net
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0: self.update_target_model()
        self.learn_step_counter = self.learn_step_counter + 1

        sample_state, sample_action, sample_reward, sample_next_state, sample_done = self.memory.sample(BATCH_SIZE)

        target_dist = self.cal_dist(sample_next_state, sample_reward, sample_done)

        eval_action = sample_action.view(-1, 1, 1).expand(-1, 1, ATOMS_NUM)  # [BATCH 1 ATOMS_NUM]
        eval_dist = self.eval_net.dist(sample_state).gather(1, eval_action).squeeze(1)
        eval_dist.data.clamp_(0.01, 0.99)
        loss = -(target_dist * eval_dist.log()).sum(1).mean()  # Wasserstein  lnp*Reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def modify_reward(state):
    x, x_dot, theta, theta_dot = state  # (位置x，x加速度, 偏移角度theta, 角加速度)
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


if __name__ == "__main__":
    agent = Categorical_DQN()
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
            if agent.memory.memory_counter >= MEMORY_CAPACITY:
                agent.learn()
                if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
                # print(agent.memory.tree.tree[-MEMORY_CAPACITY:])
            if done:
                break
            state = next_state
