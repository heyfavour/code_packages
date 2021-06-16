import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('Pendulum-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 2 2个动作
N_STATES = env.observation_space.shape[0]


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
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

    def forward(self, input):
        v = self.v_layers(input)
        q = self.q_layers(input)

        actions_value = v.expand_as(q) + (q - q.mean(1).expand_as(q))
        return actions_value


class DuelingDQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]  # return the argmax
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0]  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':

    dqn = DuelingDQN()

    print('\nCollecting experience...')
    for i_episode in range(400):
        s = env.reset()
        ep_r = 0
        acc_r = [0]
        while True:
            env.render()
            a = dqn.choose_action(s)

            f_action = (a - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # [-2 ~ 2] float actions
            # take action
            s_, reward, done, info = env.step(np.array([f_action]))

            reward /= 10  # normalize to a range of (-1, 0)
            acc_r.append(reward + acc_r[-1])  # accumulated reward
            # modify the reward

            dqn.store_transition(s, a, reward, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            s = s_
