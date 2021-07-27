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
MEMORY_CAPACITY = BATCH_SIZE = 40


class Memory(object):
    def __init__(self, memory_size=MEMORY_CAPACITY):
        # 这样写是为了需要数据保存梯度
        self.state = np.zeros(memory_size, dtype=object)
        self.action = np.zeros(memory_size, dtype=object)
        self.reward = np.zeros(memory_size, dtype=object)
        self.next_state = np.zeros(memory_size, dtype=object)
        self.done = np.zeros(memory_size, dtype=object)
        self.dist = np.zeros(memory_size, dtype=object)
        self.value = np.zeros(memory_size, dtype=object)
        self.probs = np.zeros(memory_size, dtype=object)
        self.memory_size = memory_size
        self.memory_counter = 0

    def add(self, state, action, reward, next_state, done, dist, value, probs):
        index = self.memory_counter % self.memory_size
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.done[index] = done
        self.dist[index] = dist
        self.value[index] = value
        self.probs[index] = probs
        self.memory_counter = self.memory_counter + 1

    def sample(self):
        # state, action, reward, next_state, done, dist, value, probs
        state = torch.Tensor(list(self.state))
        action = torch.Tensor(list(self.action))
        reward = torch.Tensor(list(self.reward))
        next_state = torch.Tensor(list(self.next_state))
        done = torch.Tensor(list(self.done))
        dist = np.array(self.dist)
        value = torch.Tensor(list(self.value))
        probs = torch.cat(list(self.probs))
        # print(state.size(), action.size(), reward.size(), next_state.size(), done.size(), value.size())
        # print(dist)
        # print(probs)
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
            nn.Softmax(dim=-1),
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
        self.gamma = 0.9
        self.tau = 0.95

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, value = self.ac(state)
        action = dist.sample()
        probs = dist.log_prob(action)
        return dist, action.item(), probs, value[0][0]

    def store_transition(self, state, action, reward, next_state, done, dist, value, probs):
        ##state, action, reward, next_state, done, dist, value, probs
        self.memory.add(state, action, reward, next_state, done, dist, value, probs)

    # def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    #     values = values + [next_value]
    #     gae = 0
    #     returns = []
    #     for step in reversed(range(len(rewards))):
    #         delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
    #         gae = delta + gamma * tau * masks[step] * gae
    #         returns.insert(0, gae + values[step])
    #     return returns

    def compute_age(self, next_state, reward, done, value):  # y*r-v 这里是计算y*r
        next_state = next_state[-1]
        _, _, _, next_value = self.choose_action(next_state)
        value = torch.cat((value, next_value.detach().view(-1)))
        gae = 0
        returns = []
        for step in reversed(range(BATCH_SIZE)):
            delta = reward[step] + self.gamma * value[step + 1] * (1 - done[step]) - value[step]
            gae = delta + self.gamma * self.tau * (1 - done[step]) * gae
            returns.insert(0, gae + value[step])
        #r1+0.9*v_next - v + v , (r1+0.9*v_next - v) * 0.95*0.9 + (r2+0.9*v_next - v) ,...
        returns = torch.tensor(returns)
        return returns

    def learn(self):
        state, action, reward, next_state, done, dist, value, probs = self.memory.sample()
        returns = self.compute_age(next_state, reward, done, value)
        advantage = returns - value
        actor_loss = -(probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()

        entropy = sum([d.entropy().mean() for d in dist])
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.ac_optimizer.zero_grad()
        loss.backward()
        self.ac_optimizer.step()


if __name__ == "__main__":
    agent = A2C()
    for epoch in range(2000):
        state = env.reset()
        epoch_rewards = 0
        step = 0
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
