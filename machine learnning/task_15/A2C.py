import time

import gym
import torch
import numpy as np

import torch.nn as nn
from torch.distributions import Categorical

env_name = 'CartPole-v0'
env = gym.make(env_name)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
MEMORY_CAPACITY = BATCH_SIZE = 3


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
        # state, action, reward, next_state, done,dist,value,probs
        sample = np.array(list(self.memory[:]), dtype=object).transpose()
        state = torch.FloatTensor(np.vstack(sample[0]))
        action = torch.FloatTensor(np.vstack(sample[1]))
        reward = torch.FloatTensor(list(sample[2])).view(-1)
        next_state = torch.FloatTensor(np.vstack(sample[3]))
        done = torch.FloatTensor(sample[4].astype(int)).view(-1)
        dist = sample[5]
        value = torch.FloatTensor(sample[6].astype(float)).view(-1)
        probs = sample[7]
        return state, action, reward, next_state, done, dist, value, probs

    def __len__(self):
        return self.memory_size if self.memory_counter > self.memory_size else self.memory_counter


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(N_STATES, 256),
            nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
            nn.Softmax(dim=1),
        )

        self.critic = nn.Sequential(
            nn.Linear(N_STATES, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input):
        probs = self.actor(input)
        dist = Categorical(probs)

        value = self.critic(input)
        return dist, value


class A2C:
    def __init__(self):
        self.ac = ActorCritic()
        self.ac_optimizer = torch.optim.Adam(self.ac.parameters(), lr=0.01)
        self.memory = Memory(MEMORY_CAPACITY)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.ac(state)
            action = dist.sample()
            probs = dist.log_prob(action)
        return dist, action.numpy()[0], probs, value.numpy()[0][0]

    def store_transition(self, state, action, reward, next_state, done, dist, value, probs):
        self.memory.add((state, action, reward, next_state, done, dist, value, probs))

    def compute_advantage(self,next_state, reward, done):
        _, _, _, next_value = self.choose_action(next_state[-1])
        R = next_value
        print(reward,done,R,"---------------")
        print()
        returns = []
        for step in reversed(range(BATCH_SIZE)):
            delta = reward[step] + reward * R * (1-done)[step]
            returns.insert(0, delta)
        print("==========================")
        return returns


    def learn(self):
        state, action, reward, next_state, done, dist, value, probs = self.memory.sample(BATCH_SIZE)
        returns = self.compute_advantage(next_state, reward, done)
        print(returns[0])
        print(returns)
        time.sleep(20)


if __name__ == "__main__":
    agent = A2C()
    step = 0
    for epoch in range(2000):
        state = env.reset()
        epoch_rewards = 0
        while True:
            # env.render()
            step = step + 1
            dist, action, probs, value = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done, dist, value, probs)
            if step % BATCH_SIZE == 0: agent.learn()
            state = next_state
            epoch_rewards = epoch_rewards + reward
            if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
            if done: break
