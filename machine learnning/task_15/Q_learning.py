import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# 定义参数
BATCH_SIZE = 32  # 每一批的训练量
LR = 0.01  # 学习率
EPSILON = 0.9  # 贪婪策略指数，Q-learning的一个指数，用于指示是探索还是利用。
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target的更新频率
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
N_ACTIONS = env.action_space.n  # 2 2个动作
N_STATES = env.observation_space.shape[0]  # 4 state的维度
# to confirm the shape
# 确定actiond的shape
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


# 创建神经网络模型，输出的是可能的动作

def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear": m.weight.data.normal_(0.0, 0.1)
    # m.bias.data.fill_(0)


class QNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS),
            nn.Softmax(dim=-1),

        )
        self.apply(weights_init)  # initialization

    def forward(self, input):
        output = self.model(input)
        return output


# 创建Q-learning的模型
class DQN(object):
    def __init__(self):
        # 两张网是一样的，不过就是target_net是每100次更新一次，eval_net每次都更新
        self.eval_net, self.target_net = QNet(), QNet()

        self.learn_step_counter = 0  # 如果次数到了，更新target_net
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # 选择动作
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        # input only one sample
        # np.random.uniform() 生成 0-1之间的小数
        if np.random.uniform() < EPSILON:  # 贪婪策略
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 将每个参数打包起来
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 学习过程
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach的作用就是不反向传播去更新，因为target的更新在前面定义好了的
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    dqn = DQN()
    import time

    for epoch in range(400):
        state = env.reset()  # 搜集当前环境状态。
        game_times_one_epoch = 0
        while True:
            # env.render()
            a = dqn.choose_action(state)
            time.sleep(5)
            print(a)
    #
    #         # take action
    #         s_, r, done, info = env.step(a)
    #
    #         # modify the reward
    #         x, x_dot, theta, theta_dot = s_
    #         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    #         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    #         r = r1 + r2
    #
    #         dqn.store_transition(s, a, r, s_)
    #
    #         game_times_one_epoch += r
    #         if dqn.memory_counter > MEMORY_CAPACITY:
    #             dqn.learn()
    #             if done:
    #                 print('Ep: ', epoch,'| Ep_r: ', round(ep_r, 2))
    #
    #         if done:
    #             break
    #         s = s_
