"""
value-based的算法
学习一个Q——table 存起来S=>A 不同的S采用A的得分
DQN就是用NN代替Q——table
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    #print(table)    # show table
    return table #[6 2]


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]#[1 2] [left right]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)#随机选择一个方法
    else:  # act greedy
        action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):#0 0 1
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def RL():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)  # [6,2]
    for episode in range(MAX_EPISODES):  # 13
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            # 返回下一个环境和分数，不必过于考虑这个做了什么，就是手写了一种游戏而已
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]#根据state 和 action 预测得分
            if S_ != 'terminal':
                #没结束 R+GAMMA*Q中最好的策略的最大值来更新
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
            else:
                #结束 R
                q_target = R  # next state is terminal
                is_terminated = True  # terminate this episode
            #Q = Q+ALPHA*LOSS
            q_table.loc[S, A] =q_table.loc[S, A]+ ALPHA * (q_target - q_predict)  # update q_table
            S = S_  # move to next state
            update_env(S, episode, step_counter + 1)#更新游戏环境
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = RL()
    # print('\r\nQ-table:\n')
    print(q_table)
