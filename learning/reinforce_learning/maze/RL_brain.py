"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """ 初始化

        Args:
            actions: 行为列表
            learning_rate: 学习率
            reward_decay: 奖励衰减 gamma
            e_greedy:  贪婪度

        """
        self.actions = actions  # 行为列表
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon = e_greedy  # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 初始 q_table

    def choose_action(self, observation):
        """ 选择 action

        Args:
            observation: 观察到的当前的state

        Returns:
            action: 做出判断，要采取的行动

        """
        self.check_state_exist(observation)  # 检测本 state 是否在 q_table 中存在, 不在则新建
        if np.random.uniform() < self.epsilon:  # 判断贪婪度
            state_action = self.q_table.loc[observation, :]  # 获取该状态下所有可能的 action
            # 在得分最高的action中，所及选择一个action作为最终action
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """ 根据 已知值更新 Q table

        Args:
            s: 当前 state
            a: 当前 action
            r: 当前 reward
            s_: 下一个 state

        Returns: 新 Q table

        """
        self.check_state_exist(s_)  # 检测 q_table 中是否存在 s_
        q_predict = self.q_table.loc[s, a]  # 采取当前行动后，预测得分
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        update = self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += update  # 执行更新

    def check_state_exist(self, state):
        """ 判断 state 是否存在于 Qtable, 不存在则增加"""
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

