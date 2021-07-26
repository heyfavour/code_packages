import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Parameters
env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
episodes = 20000
render = False
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.pre_model = nn.Sequential(
            nn.Linear(state_space, 32),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(32, action_space)
        self.value_head = nn.Linear(32, 1)  # Scalar Value
        self.save_actions = []
        self.rewards = []

    # Actor网络和Critic网络可以共享网络参数，两者仅最后几层使用不同结构和参数
    def forward(self, input):
        x = self.pre_model(input)
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        return F.softmax(action_score, dim=-1), state_value


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.save_actions.append(SavedAction(m.log_prob(action), state_value))  # ln[ation] value sore_memery
    return action.item()


def finish_episode():
    R = 0
    save_actions = model.save_actions
    policy_loss = []
    value_loss = []
    rewards = []  # 存放衰减处理后的分数

    for r in model.rewards[::-1]:  # 分数倒叙
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)  # 归一化

    for (log_prob, value), r in zip(save_actions, rewards):  # A R log_prob value
        reward = r - value.item()  # reward - value
        policy_loss.append(-log_prob * reward)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]


if __name__ == '__main__':
    running_reward = 10
    for epoch in range(episodes):
        state = env.reset()
        for step in count():
            env.render()
            action = select_action(state)  # 选择行为+保存记忆
            state, reward, done, info = env.step(action)
            model.rewards.append(reward)  # 保存分数

            if done: break
        running_reward = running_reward * 0.99 + step * 0.01
        print(f"Epoch:{epoch} | Step:{step}")
        # torch.save(model, modelPath)
        finish_episode()
