"""
elegant RL 可以收敛 PPO收敛稳定 但是较其他方法收敛慢
"""
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim


env = gym.make("Pendulum-v0")
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]

ENV_NAME = 'Pendulum-v0'

device = 'cpu'

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Hardswish(),
            nn.Linear(128, N_ACTIONS),
        )
        self.log_std = nn.Parameter(torch.zeros((1, 1)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        layer_norm(self.model[-1], std=0.5)  # output layer for Q value

    def forward(self, state):
        action = self.model(state).tanh()
        return action

    def get_action(self, state):
        mean = self.model(state)
        std = self.log_std.exp()
        noise = torch.randn_like(mean)
        action = mean + noise * std
        return action, noise

    # -((value - self.loc) ** 2) / (2 * (self.scale ** 2)) - math.log(self.scale) - math.log(math.sqrt(2 * math.pi))
    def get_logprob_entropy(self, state, action):
        mean = self.model(state)
        std = self.log_std.exp()

        delta = ((mean - action) / std).pow(2) * 0.5
        logprob = -(self.log_std + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.log_std + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Hardswish(),
            nn.Linear(128, 1))
        layer_norm(self.model[-1], std=0.5)  # output layer for Q value

    def forward(self, input):
        ouput = self.model(input)
        return ouput


class PPOMemory:
    def __init__(self, batch_size, batch_num):
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.memory_size = batch_size * batch_num
        self.memory = np.zeros(self.memory_size, dtype=object)
        self.memory_counter = 0

    def add(self, state, action, nosie, reward, mask):
        transition = (state, action, nosie, reward, mask)
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def sample(self):
        range = np.arange(0, self.memory_size, self.batch_size)
        indices = np.arange(self.memory_size, dtype=np.int64)
        np.random.shuffle(indices)  # 打乱样本顺序效果很差
        batch_index = [indices[i:i + self.batch_size] for i in range]
        return batch_index

    def data(self):  # state,action,nosie,reward,mask
        memory = np.array(list(self.memory), dtype=object).transpose()
        state = torch.FloatTensor(np.vstack(memory[0]))
        action = torch.FloatTensor(list(memory[1])).view(-1, 1)
        noise = torch.FloatTensor(list(memory[2])).view(-1, 1)
        reward = memory[3]
        mask = memory[4]
        return state, action, noise, reward, mask

    def __len__(self):
        return self.memory_counter


class PPO():
    def __init__(self):
        self.ratio_clip = 0.2
        self.gamma = 0.9
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.actor = Actor()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=0.0001)

        self.critic, self.critic_target = Critic(), Critic()
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=0.0001)
        self.update_target_model()

        self.batch_size = 256
        self.batch_num = 12
        self.memory = PPOMemory(self.batch_size, self.batch_num)
        self.criterion = torch.nn.SmoothL1Loss()

    def update_target_model(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actions, noises = self.actor.get_action(state)  # plan to be get_action_a_noise
        return actions[0].detach().numpy(), noises[0].detach().numpy()

    def get_reward_sum_gae(self, reward, mask, critic, next_critic):
        sum_rewards = torch.empty(self.memory.memory_size, dtype=torch.float32)  # old policy value
        advantage = torch.empty(self.memory.memory_size, dtype=torch.float32)  # advantage value
        pre_advantage = 0  # advantage value of previous step
        for i in range(self.memory.memory_size - 1, -1, -1):
            sum_rewards[i] = reward[i] + mask[i] * next_critic
            next_critic = sum_rewards[i]

            advantage[i] = reward[i] + mask[i] * (pre_advantage - critic[i])  # fix a bug here
            pre_advantage = critic[i] + advantage[i] * self.lambda_gae_adv
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)  # std
        return sum_rewards, advantage

    def soft_update(self, target_net, current_net, tau=0.1):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))

    def learn(self, next_state):
        with torch.no_grad():  # compute reverse reward
            ###########################################################
            next_critic = self.critic(torch.FloatTensor(next_state).unsqueeze(0))
            ###########################################################
            state, action, noise, reward, mask = self.memory.data()
            target_critic = self.critic_target(state).detach()

            sum_rewards, advantage = self.get_reward_sum_gae(reward, mask, target_critic, next_critic)
            log_prob = self.actor.get_old_logprob(action, noise)
        batches = self.memory.sample()
        for s in range(16):
            for batch_index in batches:
                ######################################################batch data
                batch_state = state[batch_index]
                batch_action = action[batch_index]
                batch_reward = sum_rewards[batch_index]
                batch_logprob = log_prob[batch_index]
                batch_advantage = advantage[batch_index]
                ######################################################compute loss
                new_logprob, obj_entropy = self.actor.get_logprob_entropy(batch_state, batch_action)  # it is obj_actor
                prob_ratio = (new_logprob - batch_logprob.detach()).exp()
                weighted_probs = batch_advantage * prob_ratio
                weighted_clip = batch_advantage * torch.clamp(prob_ratio, 1 - self.ratio_clip, 1 + self.ratio_clip)
                actor_loss = -torch.min(weighted_probs, weighted_clip).mean()
                #####################################################actor loss
                actor_loss = actor_loss + obj_entropy * 0.02
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                #####################################################critic loss
                new_critic = self.critic(batch_state).squeeze(1)
                critic_loss = self.criterion(new_critic, batch_reward) / (batch_reward.std() + 1e-6)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                ####################################################soft update
                self.soft_update(self.critic_target, self.critic)


if __name__ == '__main__':
    agent = PPO()
    epoch_list = []
    for epoch in range(20000):
        state = env.reset()
        epoch_rewards = 0
        while True:
            action, noise = agent.choose_action(state)
            next_state, reward, done, _ = env.step(2.0*np.tanh(action))
            # state action nosie reward mask
            agent.memory.add(state, action, noise, reward * 0.125, 0.0 if done else 0.99)
            epoch_rewards = epoch_rewards + reward
            state = next_state
            if len(agent.memory) % agent.memory.memory_size == 0: agent.learn(next_state)
            if done: break
        epoch_list.append(epoch_rewards)
        print(f"Episode:{epoch:0=3d}, Reward:{epoch_rewards:.2f} Mean:{sum(epoch_list[-10:]) / 10:.2f} STD:{agent.actor.log_std.item():.3f}")
