"""
相当稳定的一种算法
收敛比DQN稳定 400step分数突变过万（哭泣，DQN可以快速收敛到2000-3000但是相当不稳定,以后再也不用DQN了）
吐槽一句理解了好久，没有DQN好理解
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
    def __init__(self, memery_size, batch_step):
        self.memory = np.zeros(memery_size, dtype=object)
        self.memory_size = memery_size
        self.batch_step = batch_step

        self.memory_counter = 0

    def add(self, transition):  # (state, action, reward, done, prob, critic)
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def sample(self):
        # 每100步算一个epoch 每一个epoch更新memory_size/batch_step 次
        min_batch_range = np.arange(0, self.memory_size, self.batch_step)
        indices = np.arange(self.memory_size, dtype=np.int64)
        np.random.shuffle(indices)
        batch_index = [indices[i:i + self.batch_step] for i in min_batch_range]
        return batch_index

    def data(self):  # (state, action, reward, done, prob, critic)
        memory = np.array(list(self.memory), dtype=object).transpose()
        state = torch.FloatTensor(np.vstack(memory[0]))
        action = torch.LongTensor(list(memory[1])).view(-1, 1)
        reward = torch.FloatTensor(list(memory[2])).view(-1, 1)
        done = torch.FloatTensor(memory[4].astype(int)).view(-1, 1)
        prob = torch.FloatTensor(list(memory[5])).view(-1, 1)
        critic = torch.FloatTensor(list(memory[6])).view(-1, 1)
        return state, action, reward, done, prob, critic


class PPO():
    def __init__(self):
        self.policy_clip = 0.2
        self.epoch = 5
        self.gae_lambda = 0.95
        self.actor = Actor()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=0.0005)

        self.critic = Critic()
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=0.0005)

        self.memory_size = 100
        self.batch_step = 10#每个batch多少步
        self.memory = PPOMemory(self.memory_size, self.batch_step)

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            state = torch.FloatTensor(state)
            # action
            action_dist = self.actor(state)
            action = action_dist.sample()
            action_prob = action_dist.log_prob(action)
            # value
            critic = self.critic(state)
        return action.item(), action_prob.item(), critic.item()

    def compute_advantage(self, reward, critic, done):  # GAE方法获取returns
        advantage = np.zeros(self.memory.memory_size, dtype=np.float32)  # [batch_size]
        for i in range(self.memory.memory_size - 1):#[0 1 2 3 4 5 6 7 8 9]
            discount, returns = 1, 0
            for step in range(i, self.memory.memory_size - 1):#[0 1 2 3 4 5 6 7 8 9]
                # Q = Q + d*(R[STEP] + G*V[STEP+1]*(1-D) - V[K])
                delta = reward[step] + GAMMA * critic[step + 1] * (1 - done[step]) - critic[step]
                returns = returns + discount * delta
                discount = discount * GAMMA * self.gae_lambda  # D*0.99*0.95
            advantage[i] = returns#return = return + (G*GAE)^n*DELTA
        advantage = torch.tensor(advantage)
        return advantage

    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae #delta + G * 0.95(1-DONE)*0.
            returns.insert(0, gae + values[step])
        return returns

    def learn(self):
        for _ in range(self.epoch):
            ### compute advantage ###
            #state, action, reward, done, prob, critic = self.memory.data()
            advantage = self.compute_advantage(reward, critic, done)  # a = a + d*(r + (1-D)*G*V[K+1] - V[K])
            batch = self.memory.sample()
            print(batch)

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

                # critic_loss = F.smooth_l1_loss(advantage[batch], value.squeeze())
                critic_loss = ((advantage[batch] + values[batch] - critic_value) ** 2).mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()


if __name__ == '__main__':
    agent = PPO()
    step = 0
    for epoch in range(20000):
        state = env.reset()
        epoch_rewards = 0
        while True:
            step = step + 1
            action, prob, critic = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.add((state, action, reward, done, prob, critic))

            if step % agent.memory.memory_size == 0: agent.learn()

            state = next_state
            epoch_rewards = epoch_rewards + reward
            if done: break

        print(f"Episode:{epoch:0=3d}, Reward:{epoch_rewards:.3f}")
