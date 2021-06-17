"""
感觉收敛速度没有double DQN快
"""
import torch
import torch.nn as nn
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 1000
# env = gym.make('Pendulum-v0')
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n  #
N_STATES = env.observation_space.shape[0]


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear": m.weight.data.normal_(0.0, 0.1)


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.v_layers = nn.Sequential(
            nn.Linear(N_STATES, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.q_layers = nn.Sequential(
            nn.Linear(N_STATES, 10),
            nn.ReLU(),
            nn.Linear(10, N_ACTIONS),
        )
        self.apply(weights_init)  # initialization

    def forward(self, input):
        v = self.v_layers(input)  # 环境的价值
        q = self.q_layers(input)  # 动作的价值
        # actions_value = v.expand_as(q) + (q - q.mean(dim=1,keepdim=True).expand_as(q)) 等同与下方
        actions_value = v + (q - q.mean(dim=1, keepdim=True))
        return actions_value


class DuelingDQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        if np.random.uniform() < EPSILON:  # greedy
            action = self.predict_action(state)
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def predict_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            actions_value = self.eval_net(state)  # [1 2]
            action = actions_value.max(1)[1].item()
            return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))  # 将每个参数打包起来 4+1+1+4  [10]
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter = self.memory_counter + 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = self.learn_step_counter + 1

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


if __name__ == '__main__':
    dqn = DuelingDQN()
    for epoch in range(500):
        state = env.reset()  # 搜集当前环境状态。
        epoch_rewards = 0
        while True:
            # env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state  # (位置x，x加速度, 偏移角度theta, 角加速度)
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            epoch_rewards = epoch_rewards + reward
            dqn.store_transition(state, action, reward, next_state)
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
            if done:
                break
            state = next_state
        if epoch_rewards > 500: break

    dqn.eval_net.eval()
    with torch.no_grad():
        state = env.reset()  # 搜集当前环境状态。
        epoch_rewards = 0
        while True:
            # env.render()
            action = dqn.predict_action(state)
            next_state, reward, done, info = env.step(action)
            # modify the reward 如果不重定义分数，相当难收敛
            epoch_rewards = epoch_rewards + reward
            if done:
                print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
                break
            state = next_state
