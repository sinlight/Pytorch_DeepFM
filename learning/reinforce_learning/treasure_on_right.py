# treasure_on_right.py

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

N_STATES = 6  #  1维度状态空间大小
ACTIONS = ['left', 'right']  # available actions  动作空间
EPSILON = 0.9  # greedy police 百分之90的情况我会按照Q表的最优值选择行为，百分之10的时间随机选择行为
ALPHA = 0.1  # learning rate 决定这次的误差有多少是要被学习的
GAMMA = 0.9  # discount factor 对未来reward的衰减值。gamma越接近1，机器对未来的reward越敏感
MAX_EPISODES = 13  # maximum episodes 最大回合数
FRESH_TIME = 0.3  # fresh time for one move 移动间隔时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)  # 随机选择行动
    else:
        # act greedy
        # replace argmax to idxmax as argmax means a different function in newer version of pandas
        action_name = state_actions.idxmax()  # 返回series最大值的索引
    return action_name


def get_env_feedback(S, A):
    """ 环境交互
    Args:
        S -- situation 当前状态
        A -- action 执行操作
    Returns:
        S_ -- 下一步状态
        R -- 操作有效性反馈
    """
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


def update_env(S, episode, step_counter):
    # 用于展现当前状态 T为宝藏位置，o为探索者位置
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S == 'terminal':
        interaction = '迭代次数 %s: 总步数 = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始化 q table
    for episode in range(MAX_EPISODES):  # 回合
        step_counter = 0
        S = 0  # 回合初始的位置
        is_terminated = False
        print('回合', episode)
        update_env(S, episode, step_counter)  # 环境更新
        while not is_terminated:

            A = choose_action(S, q_table)  # 根据当前状态选择 action
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A]  # 估算的（状态-行为）值，q_table 代表预测值
            if S_ != 'terminal':  # next state is not terminal
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # 实际的（状态-行为)值
                # Q(s,a) = Q(s,a) + alpha[r + gamma*max(Q(s',a')|s') - Q(s,a)]
            else:  # next state is terminal
                q_target = R  # 实际的（状态-行为值）
                is_terminated = True  # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # 更新 q_table
            S = S_  # 更新探索者位置

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
