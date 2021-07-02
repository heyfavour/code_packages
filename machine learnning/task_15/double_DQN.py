"""
double DQN 使用了target net
"""
import torch
import torch.nn as nn
import numpy as np
import gym

# 定义参数
BATCH_SIZE = 64  # 每一批的训练量
LR = 0.0005  # 学习率
EPSILON = 0.9  # 贪婪策略指数，Q-learning的一个指数，用于指示是探索还是利用。
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target的更新频率
MEMORY_CAPACITY = 5000  # memery较小时，训练较快，但是后期效果没记忆体大好，建议动态记忆体，前期记忆体小，后期记忆体大

env = gym.make('CartPole-v0')
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
N_ACTIONS = env.action_space.n  # 2 2个动作
N_STATES = env.observation_space.shape[0]  # 4 state的维度
# to confirm the shape
# 确定action的shape
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


# 创建神经网络模型，输出的是可能的动作

def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


class Q_TABLE_NET(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.ReLU(),
            nn.Linear(50, 50),  # 2层隐藏层以后效果更好 一层时，到200d多epoch才收敛
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS),
            # nn.Softmax(dim=-1), 如果加了很难收敛
        )
        # self.apply(weights_init)  # initialization

    def forward(self, input):
        output = self.model(input)
        return output


# 创建Q-learning的模型
class DQN_Agent(object):
    def __init__(self):
        # 两张网是一样的，不过就是target_net是每100次更新一次，eval_net每次都更新
        self.eval_net, self.target_net = Q_TABLE_NET(), Q_TABLE_NET()

        self.learn_step_counter = 0  # 如果次数到了，更新target_net
        self.memory_counter = 0  # for storing memory
        # 初始化记忆 [2000 10] 10=state+ action+reward+next_state
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.AdamW(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # 选择动作
    def choose_action(self, state):
        # np.random.uniform() 生成 0-1之间的小数
        if np.random.uniform() < EPSILON:  # 贪婪策略
            action = self.predict(state)
        else:  # random
            action = np.random.randint(0, N_ACTIONS)  # 随机产生一个action
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # 如果action是多维度的则变形
        return action

    def predict(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            # actions_value = self.eval_net.forward(state)#[1 2] 莫凡pyhon中使用forward不是很理解
            actions_value = self.eval_net(state)  # [1 2]
            # torch.max(actions_value) 返回最大的value
            # torch.max(actions_value,1) 返回最大的(value index)
            action = actions_value.max(1)[1].item()
            #action = torch.max(actions_value, 1)[1].data.numpy()  # return the argmax index
            #action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # 如果action是多维度的则变形
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

def modify_reward(state):
    x, x_dot, theta, theta_dot = state  # (位置x，x加速度, 偏移角度theta, 角加速度)
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


if __name__ == '__main__':
    dqn = DQN_Agent()

    for epoch in range(80000):
        state = env.reset()  # 搜集当前环境状态。
        epoch_rewards = 0
        while True:
            # env.render()
            action = dqn.choose_action(state)
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
    # eval
    dqn.eval_net.eval()
    with torch.no_grad():
        state = env.reset()  # 搜集当前环境状态。
        epoch_rewards = 0
        while True:
            #env.render()
            action = dqn.predict(state)
            next_state, reward, done, info = env.step(action)
            # modify the reward 如果不重定义分数，相当难收敛
            # reward = modify_reward(next_state)
            epoch_rewards = epoch_rewards + reward

            if done:
                print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
                break
            state = next_state
