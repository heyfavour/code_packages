# !/usr/bin/env python
# coding=utf-8
import gym
import numpy as np
import torch
import torch.optim as optim

from torch import nn
from torch.distributions import Normal, MultivariateNormal

env = gym.make('Pendulum-v0')
env.unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]


class Actor(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(1, N_ACTIONS) * 0.01)

    def forward(self, state):
        mu = self.mu(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        # std = torch.diag(self.log_std)
        # dist = MultivariateNormal(mu, std)
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
    def __init__(self, batch_size, batch_num):
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.memory_size = batch_size * batch_num
        self.memory = np.zeros(self.memory_size, dtype=object)
        self.memory_counter = 0

    def add(self, state, action, prob, critic, reward, done):
        # [0.02438012 0.01726248 0.10023742 0.3512741 ] 0 -1.1162703037261963 17.281251907348633 1.0 False
        transition = (state, action, prob, critic, reward, done)
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def sample(self):
        range = np.arange(0, self.memory_size, self.batch_size)
        indices = np.arange(self.memory_size, dtype=np.int64)
        np.random.shuffle(indices)
        batch_index = [indices[i:i + self.batch_size] for i in range]
        return batch_index

    def data(self):  # (state, action, prob, critic, reward, done)
        memory = np.array(list(self.memory), dtype=object).transpose()
        state = torch.FloatTensor(np.vstack(memory[0]))
        action = torch.LongTensor(np.vstack(memory[0])).view(-1)
        prob = torch.FloatTensor(list(memory[2])).view(-1)
        critic = torch.FloatTensor(list(memory[3])).view(-1)
        reward = torch.FloatTensor(list(memory[4])).view(-1, 1)
        done = torch.FloatTensor(memory[5].astype(int)).view(-1)
        return state, action, prob, critic, reward, done


class PPO():
    def __init__(self):
        self.policy_clip = 0.2
        self.gamma = 0.9
        self.gae_lambda = 0.95
        self.actor = Actor()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=0.005)

        self.critic = Critic()
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=0.005)

        self.batch_size = 10
        self.batch_num = 4
        self.memory = PPOMemory(self.batch_size, self.batch_num)

    def choose_action(self, observation):
        state = torch.unsqueeze(torch.FloatTensor(observation), 0)  # [4]=>[1 4]
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.sample().clip(-2.0,2.0)
            probs = dist.log_prob(action)
            value = self.critic(state)
        return action.numpy()[0], probs.numpy()[0], value.item()

    def compute_advantage(self, reward, values, dones):
        advantage_len = self.memory.memory_size
        advantage = np.zeros(advantage_len, dtype=np.float32)

        for t in range(advantage_len - 1):
            discount = 1
            adv = 0
            for k in range(t, advantage_len - 1):
                delta = reward[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k]
                adv = adv + discount * delta
                discount = discount * self.gamma * self.gae_lambda
            advantage[t] = adv
        advantage = torch.tensor(advantage)
        return advantage

    def mini_batch_loss(self, batch_state, batch_prob, batch_action, batch_critic, batch_advantage):
        dist = self.actor(batch_state)
        entropy = dist.entropy().mean()
        critic_value = self.critic(batch_state).squeeze()

        new_probs = dist.log_prob(batch_action)
        prob_ratio = new_probs.exp() / batch_prob.exp()
        weighted_probs = batch_advantage * prob_ratio
        weighted_clip = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantage
        actor_loss = -torch.min(weighted_probs, weighted_clip).mean()
        returns = batch_advantage + batch_critic
        critic_loss = (returns - critic_value) ** 2 - 0.001 * entropy
        critic_loss = critic_loss.mean()
        total_loss = actor_loss + 0.5 * critic_loss
        return total_loss

    def learn(self):
        batches = self.memory.sample()
        state, action, prob, critic, reward, done = self.memory.data()
        ### compute advantage ###
        advantage = self.compute_advantage(reward, critic, done)
        ### SGD ###
        for batch in batches:
            batch_state = state[batch]
            batch_prob = prob[batch]
            batch_action = action[batch]
            batch_advantage = advantage[batch]
            batch_critic = critic[batch]

            total_loss = self.mini_batch_loss(batch_state, batch_prob, batch_action, batch_critic, batch_advantage)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == '__main__':
    agent = PPO()
    step = 0
    for epoch in range(20000):
        state = env.reset()
        # env.render()
        epoch_rewards = 0
        while True:
            step = step + 1
            action, prob, critic = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.add(state, action, prob, critic, reward, done)

            if step % agent.memory.memory_size == 0: agent.learn()

            state = next_state
            epoch_rewards = epoch_rewards + reward
            if done: break

        print(f"Episode:{epoch:0=3d}, Reward:{epoch_rewards:.3f}")
