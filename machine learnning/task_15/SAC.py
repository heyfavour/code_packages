#!/usr/bin/env python
# coding=utf-8
import time

import gym
import torch
import numpy as np

from torch import nn
from torch.distributions import Normal

env = gym.make('Pendulum-v0')
# env = env.unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]

GAMMA = 0.99
MEMORY_CAPACITY = 1000
BATCH_SIZE = 64


class Memory(object):
    def __init__(self, memory_size=MEMORY_CAPACITY):
        self.memory = np.zeros(memory_size, dtype=object)
        self.memory_size = memory_size
        self.memory_counter = 0

    def add(self, transition):  # data = (state, action, reward, next_state, done) 4 1 1 4 1
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def sample(self, batch_size=BATCH_SIZE):
        sample_index = np.random.choice(len(self), batch_size)  # 2000中取一个batch index [batch_size]
        # [(s a r s) ] -> [(s) (a) (r) (s)]
        sample_memory = np.array(list(self.memory[sample_index]), dtype=object).transpose()
        sample_state = torch.FloatTensor(np.vstack(sample_memory[0]))
        sample_action = torch.FloatTensor(np.vstack(sample_memory[1]))
        sample_reward = torch.FloatTensor(list(sample_memory[2])).view(-1, 1)
        sample_next_state = torch.FloatTensor(np.vstack(sample_memory[3]))
        sample_done = torch.FloatTensor(sample_memory[4].astype(int)).view(-1, 1)
        return sample_state, sample_action, sample_reward, sample_next_state, sample_done

    def __len__(self):
        return self.memory_size if self.memory_counter > self.memory_size else self.memory_counter


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(N_STATES, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state):
        output = self.critic(state)
        return output


class SoftNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft = nn.Sequential(
            nn.Linear(N_STATES + N_ACTIONS, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        input = torch.cat([state, action], 1)
        output = self.soft(input)
        return output


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, N_ACTIONS)
        self.log_std = nn.Linear(256, N_ACTIONS)

    def forward(self, state):
        mid = self.model(state)
        mean = self.mean(mid)
        log_std = self.log_std(mid).clamp(-2, 2)
        return mean, log_std


class SAC:
    def __init__(self) -> None:
        self.action_range = [env.action_space.low, env.action_space.high]
        self.memory = Memory(MEMORY_CAPACITY)
        # NET
        self.critic, self.target_critic = Critic(), Critic()
        self.soft_1,self.soft_2 = SoftNet(),SoftNet()#TD3
        self.actor = Actor()
        self.update_target_model()
        # optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.soft_1_optimizer = torch.optim.Adam(self.soft_1.parameters(), lr=0.001)
        self.soft_2_optimizer = torch.optim.Adam(self.soft_2.parameters(), lr=0.001)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()

        self.soft_tau = 0.01  # 控制更新得幅度

    def update_target_model(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_model(self):
        # w = t*w *(1-t)*w
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(state)
            normal = Normal(mean, log_std.exp())
            z = normal.sample()
            action = torch.tanh(z).detach().numpy()[0] * 2.0
        return action

    def epsilon_action(self, state, epsilon=1e-6):
        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z)
        log_prob = normal.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action*2.0, log_prob


    def learn(self):
        if len(self.memory) < BATCH_SIZE: return

        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        new_action, log_prob= self.epsilon_action(state)#actor
        #critic loss
        critic = self.critic(state)
        new_soft_1 = self.soft_1(state, new_action)
        new_soft_2 = self.soft_2(state, new_action)
        next_critic = torch.min(new_soft_1, new_soft_2) - log_prob
        critic_loss  =self.loss_func(critic, next_critic.detach())#critic min(soft)-logP
        #soft loss
        soft_1 = self.soft_1(state, action)
        soft_2 = self.soft_2(state, action)
        target_critic = self.target_critic(next_state)
        target_Q = reward + (1 - done) * GAMMA * target_critic

        soft_1_loss = self.loss_func(soft_1, target_Q.detach())
        soft_2_loss = self.loss_func(soft_2, target_Q.detach())
        #actor loss
        actor_loss = (log_prob - torch.min(new_soft_1, new_soft_2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.soft_1_optimizer.zero_grad()
        self.soft_2_optimizer.zero_grad()
        soft_1_loss.backward()
        soft_2_loss.backward()
        self.soft_1_optimizer.step()
        self.soft_2_optimizer.step()



        self.soft_update_model()

if __name__ == "__main__":
    agent = SAC()
    for epoch in range(2000):
        state = env.reset()
        epoch_rewards = 0
        while True:
            # env.render()
            action = agent.choose_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state

            epoch_rewards = epoch_rewards + reward
            if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
            if done: break
