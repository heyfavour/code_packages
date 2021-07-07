#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 64
GAMMA = 0.9  # reward discount
MEMORY_CAPACITY = 2000

env = gym.make('Pendulum-v0')
env.unwrapped
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]


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
        return min(self.memory_size, self.memory_counter)


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.actor(input)
        return output


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        # Q1 architecture
        self.critic_Q1 = nn.Sequential(
            nn.Linear(N_STATES + N_ACTIONS, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

        # Q2 architecture
        self.critic_Q2 = nn.Sequential(
            nn.Linear(N_STATES + N_ACTIONS, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, state, action):
        input = torch.cat([state, action], 1)  # 按维数1拼接
        q_1 = self.critic_Q1(input)
        q_2 = self.critic_Q2(input)
        return q_1, q_2


class OUNoise(object):  # Ornstein–Uhlenbeck
    # 适合 惯性系统，尤其是时间离散化粒度较小
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high

        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

        self.reset()

    def reset(self):  # [1 1 1 1]*0.0
        self.obs = np.ones(self.action_dim) * self.mu

    def evolve_obs(self):
        # OBS + THETA*(MU-OBS)+SIGMA*RANDOM_ACTION
        self.obs = self.obs + self.theta * (self.mu - self.obs) + self.sigma * np.random.randn(self.action_dim)
        return self.obs

    def get_action(self, action):
        ou_obs = self.evolve_obs()
        # sigma = max-(max-min)*min(1,t/10000)
        # self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)  # 动态 sigma
        return np.clip(action + ou_obs, self.low, self.high)


class TD3(object):
    def __init__(self):
        self.gamma = 0.9
        self.soft_tau = 0.01  # 控制更新得幅度
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 5  # policy更新频率
        self.exploration_noise = 0.1
        self.max_action = 2.0
        self.min_action = -2.0

        self.actor, self.target_actor = Actor(), Actor()
        self.critic, self.target_critic = Critic(), Critic()

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=0.0003)
        self.update_target_model()

        self.memory = Memory(MEMORY_CAPACITY)
        self.loss_func = nn.MSELoss()

    def update_target_model(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_model(self):
        # w = t*w *(1-t)*w
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def choose_action(self, state):
        action = self.predict_action(state)
        action = action + np.random.normal(0, self.exploration_noise, size=N_ACTIONS)
        action = action.clip(self.min_action, self.max_action)
        return action

    def predict_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).detach().numpy()[0] * 2.0
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def critic_learn(self, state, action, reward, next_state, done):
        with torch.no_grad():
            # randn_like 返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
            # noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            noise = torch.ones_like(action).data.normal_(0, self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.target_actor(next_state) + noise).clamp(self.min_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        # Compute critic loss
        critic_loss = self.loss_func(current_Q1, target_Q) + self.loss_func(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def actor_learn(self, state):
        Q1, Q2 = self.critic(state, self.actor(state))
        actor_loss = -torch.min(Q1, Q2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def learn(self):
        if len(self.memory) < BATCH_SIZE: return
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        self.critic_learn(state, action, reward, next_state, done)
        # Delayed policy updates
        if self.memory.memory_counter % self.policy_delay == 0:
            self.actor_learn(state)
            self.soft_update_model()


if __name__ == '__main__':
    agent = TD3()
    ou_noise = OUNoise(env.action_space)
    for epoch in range(2000):
        state = env.reset()
        epoch_rewards = 0
        ou_noise.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            epoch_rewards = epoch_rewards + reward
            if done:
                print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
                break
