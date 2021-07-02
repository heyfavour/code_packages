"""
相当稳定的一种算法
收敛比DQN稳定 400step分数突变过万（哭泣，DQN可以快速收敛到2000-3000但是相当不稳定,以后再也不用DQN了）
"""
# !/usr/bin/env python
# coding=utf-8
import gym
import numpy as np
import torch
import torch.optim as optim

from torch import nn
from torch.distributions.categorical import Categorical

env = gym.make("CartPole-v0")
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
N_ACTIONS = env.action_space.n  # 2 2个动作
N_STATES = env.observation_space.shape[0]  # 4 state的维度
GAMMA = 0.99

class Actor(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class Critic(nn.Module):
    def __init__(self, ):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)  # [0 S 32步长]
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_step]  # batch_step个 batch_size大小得随机数组
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(
            self.rewards), np.array(self.dones), batches

    def add(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPO():
    def __init__(self):
        self.policy_clip = 0.2
        self.epoch = 5
        self.gae_lambda = 0.95
        self.actor = Actor()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=0.0005)

        self.critic = Critic()
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=0.0005)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.batch_size = 64
        self.memory = PPOMemory(self.batch_size)

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        # action
        dist = self.actor(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        # value
        value = self.critic(state)
        value = torch.squeeze(value).item()
        return action, probs, value

    def compute_advantage(self, reward, value, done):
        advantage = np.zeros(self.batch_size, dtype=np.float32)  # [batch_size]
        for i in range(self.batch_size - 1):
            discount, Q = 1, 0
            for k in range(i, self.batch_size - 1):
                # a = a + d*(r + (1-D)*G*V[K+1] - V[K])
                Q = Q + discount * (reward[k] + GAMMA * value[k + 1] * (1 - done[k]) - value[k])
                discount = discount * GAMMA * self.gae_lambda  # D*0.99*0.95
            advantage[i] = Q
        advantage = torch.tensor(advantage)
        return advantage

    def learn(self):
        for _ in range(self.epoch):
            state, action, prob, value, reward, done, batch = self.memory.sample()
            ### compute advantage ###
            advantage = self.compute_advantage(reward, value, done)  # a = a + d*(r + (1-D)*G*V[K+1] - V[K])
            ### SGD ###
            values = torch.tensor(value)
            # train_batch()
            for batch in batch:  # [[ 0,  6,  7,  3, 14],[ 5, 16,  8, 13, 11],...]
                states = torch.tensor(state[batch], dtype=torch.float)
                old_probs = torch.tensor(prob[batch])
                actions = torch.tensor(action[batch])

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()
                action_prob = dist.log_prob(actions)  # action_prob

                prob_ratio = action_prob.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio  # [a + d*(r + (1-D)*G*V[K+1] - V[K]) ] * P1/p2
                clip_prob = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, clip_prob).mean()

                critic_loss = ((advantage[batch] + values[batch] - critic_value) ** 2).mean()
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()


def modify_reward(state):
    x, x_dot, theta, theta_dot = state  # (位置x，x加速度, 偏移角度theta, 角加速度)
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


if __name__ == '__main__':
    agent = PPO()
    step = 0
    for epoch in range(20000):
        state = env.reset()
        epoch_rewards = 0
        while True:
            # if epoch>200:env.render()
            step = step + 1
            action, prob, val = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            # reward = modify_reward(next_state)
            agent.memory.add(state, action, prob, val, reward, done)
            # frequency of agent update
            if step % 64 == 0: agent.learn()

            state = next_state
            epoch_rewards = epoch_rewards + reward
            if done: break

        print(f"Episode:{epoch:0=3d}, Reward:{epoch_rewards:.3f}")
