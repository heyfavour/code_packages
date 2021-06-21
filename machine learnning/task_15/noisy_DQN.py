"""
没感觉有啥用
"""
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 64  # 每一批的训练量
LR = 0.001  # 学习率
EPSILON = 0.9  # 贪婪策略指数，Q-learning的一个指数，用于指示是探索还是利用。
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target的更新频率
MEMORY_CAPACITY = 1000  # memery较小时，训练较快，但是后期效果没记忆体大好，建议动态记忆体，前期记忆体小，后期记忆体大

env = gym.make('CartPole-v0')
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
N_ACTIONS = env.action_space.n  # 2 2个动作
N_STATES = env.observation_space.shape[0]  # 4 state的维度


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear": m.weight.data.normal_(0.0, 0.1)
    # m.bias.data.fill_(0)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))  # [10 10]
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.out_features, self.in_features))

        self.bias = nn.Parameter(torch.FloatTensor(self.out_features))
        self.bias_sigam = nn.Parameter(torch.FloatTensor(self.out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.out_features))
        self.reset_parameter()
        self.reset_noise()

    def reset_parameter(self):
        mu_range = 1. / np.sqrt(self.in_features)  # 1/平方根 input_dum=2->1/1.414
        self.weight.detach().uniform_(-mu_range, mu_range)  # w->随机入mu_range的随机数初始化
        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.in_features))  # mu_range*self.std_init

        self.bias.detach().uniform_(-mu_range, mu_range)
        self.bias_sigam.detach().fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))  # 外积[out]外积[in] [out_size in_size]
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))  # [out_size]

    def _scale_noise(self, size):
        noise = torch.randn(size)  # [-1 1]的正态分布
        noise = noise.sign() * noise.abs().sqrt()  # [noise的根号]
        return noise

    def forward(self, input):
        if self.training:
            weight = self.weight + self.weight_sigma * self.weight_epsilon
            bias = self.bias + self.bias_sigam * self.bias_epsilon
        else:
            weight = self.weight
            bias = self.bias
        return F.linear(input, weight, bias)


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.noisy_layer_mid = NoisyLinear(10, 10)
        self.noisy_layer_out = NoisyLinear(10, N_ACTIONS)

        self.model = nn.Sequential(
            nn.Linear(N_STATES, 10),
            nn.ReLU(),
            self.noisy_layer_mid,
            nn.ReLU(),
            self.noisy_layer_out
        )
        self.apply(weights_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.model(input)
        return output

    def reset_noise(self):
        self.noisy_layer_mid.reset_noise()
        self.noisy_layer_out.reset_noise()


# 创建Q-learning的模型
class DQN_Agent(object):
    def __init__(self):
        # 两张网是一样的，不过就是target_net是每100次更新一次，eval_net每次都更新
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 如果次数到了，更新target_net
        self.memory_counter = 0  # for storing memory
        # 初始化记忆 [2000 10] 10=state+ action+reward+next_state
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.AdamW(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        if np.random.uniform() < EPSILON:  # 贪婪策略
            action = self.predict_action(state)
        else:
            action = np.random.randint(0, N_ACTIONS)  # 随机产生一个action
        return action

    def predict_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            actions_value = self.eval_net(state)  # [1 2]
            action = actions_value.max(1)[1].item()
            return action

    # 存储记忆
    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))  # 将每个参数打包起来 4+1+1+4  [10]
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter = self.memory_counter + 1
        # print(self.memory_counter)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # TARGET_REPLACE_ITER=100 target 更新的频率
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.eval_net.reset_noise()
            self.target_net.reset_noise()

        self.learn_step_counter = self.learn_step_counter + 1

        # 学习过程
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 2000中取一个batch index [batch_size]
        sample_memory = self.memory[sample_index, :]
        sample_state = torch.FloatTensor(sample_memory[:, :N_STATES])  # [64 4]
        sample_action = torch.LongTensor(sample_memory[:, N_STATES:N_STATES + 1].astype(int))  # [batch 1]
        sample_reward = torch.FloatTensor(sample_memory[:, N_STATES + 1:N_STATES + 2])  # [batch 1]
        sample_next_state = torch.FloatTensor(sample_memory[:, -N_STATES:])  # [batch 4]

        # q_eval w.r.t the action in experience
        # input [batch_size dim_state] => output (batch, 2) [32,2]=>[32 1]=[batch_size action]
        q_eval = self.eval_net(sample_state).gather(1, sample_action)  # 去对应的acion的实际output
        # detach的作用就是不反向传播去更新，因为target的更新在前面定义好了的 [batch_size action]
        q_next = self.target_net(sample_next_state).detach()
        # Q = r + GAMMA*MAX(Q)
        q_target = sample_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1) TD MC?
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.eval_net.reset_noise()
        # self.target_net.reset_noise()


# modify the reward 如果不重定义分数，相当难收敛
def modify_reward(state):
    x, x_dot, theta, theta_dot = state  # (位置x，x加速度, 偏移角度theta, 角加速度)
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


if __name__ == '__main__':
    dqn = DQN_Agent()

    for epoch in range(400):
        state = env.reset()  # 搜集当前环境状态。
        epoch_rewards = 0
        while True:
            # env.render()
            action = dqn.choose_action(state)
            # action = dqn.predict_action(state)
            # take action
            next_state, reward, done, info = env.step(action)
            reward = modify_reward(next_state)
            epoch_rewards = epoch_rewards + reward

            dqn.store_transition(state, action, reward, next_state)

            if dqn.memory_counter > MEMORY_CAPACITY:  # 记忆超过2000次后
                dqn.learn()
                if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
            if done:
                break
            state = next_state
