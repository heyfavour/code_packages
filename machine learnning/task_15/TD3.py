#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 256
GAMMA = 0.9  # reward discount
MEMORY_CAPACITY = 5000

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
        return self.memory_size if self.memory_counter > self.memory_size else self.memory_counter


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.Linear(256, 256)
            nn.Linear(256, N_ACTIONS)
        )

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class NormalizedActions(gym.ActionWrapper):  # 将action范围重定在[0.1]之间
    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action


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

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        # sigmae = max-(max-min)*min(1,t/10000)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)  # 动态 sigma
        return np.clip(action + ou_obs, self.low, self.high)


class DDPG():
    def __init__(self, device="cpu"):
        self.device = device
        self.actor, self.target_actor = Actor(), Actor()
        self.critic, self.target_critic = Critic(), Critic()
        self.update_target_model()

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=0.001)

        self.memory = Memory(MEMORY_CAPACITY)
        self.loss_func = nn.MSELoss()

        self.soft_tau = 0.01  # 控制更新得幅度

    def update_target_model(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_model(self):
        # w = t*w *(1-t)*w
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def choose_action(self, state):  # 由于使用了其他噪声，所以不需要epsilon 贪婪策略
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad(): action = self.actor(state).detach().numpy()[0]
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def actor_learn(self, state):
        loss = self.critic(state, self.actor(state))  # state,action
        loss = -1 * loss.mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def critic_learn(self, state, action, reward, next_state, done):
        q_eval = self.critic(state, action)

        next_action = self.target_actor(next_state).detach()
        q_next = self.target_critic(next_state, next_action).detach()
        q_target = reward + (1.0 - done) * GAMMA * q_next
        # q_target = torch.clamp(q_target, -np.inf, np.inf)
        loss = self.loss_func(q_eval, q_target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def learn(self):
        if len(self.memory) < BATCH_SIZE: return
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        self.actor_learn(state)
        self.critic_learn(state, action, reward, next_state, done)

        self.soft_update_model()  # 软更新


class TD3(object):
    def __init__(self,state_dim,action_dim,max_action, cfg,):
        self.max_action = max_action
        self.gamma = 0.99
        self.lr = 0.0005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 256
        self.total_it = 0

        self.actor = Actor()
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.memory = ReplayBuffer(state_dim, action_dim)

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.memory.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.lr * param.data + (1 - self.lr) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.lr * param.data + (1 - self.lr) * target_param.data)



if __name__ == '__main__':
    env = NormalizedActions(env)
    agent = DDPG()
    ou_noise = OUNoise(env.action_space)  # env.action_space  Box(-2.0, 2.0, (1,), float32) min max shape

    for epoch in range(2000):
        state = env.reset()
        epoch_rewards = 0
        ou_noise.reset()
        step = 0
        while True:
            env.render()
            step = step + 1
            action = agent.choose_action(state)
            # action = np.clip(np.random.normal(action,NOISE),-1.0,1.0) #guassion noise
            if epoch < 1000:action = ou_noise.get_action(action, step)  # 即paper中的random process
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            epoch_rewards = epoch_rewards + reward
            if done:
                if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
                break
