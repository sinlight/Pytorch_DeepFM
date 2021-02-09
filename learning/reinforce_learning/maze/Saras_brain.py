"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class RL(object):
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
        """ 选择 action 相当于预测

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

    def learn(self, *args):
        pass


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """ 初始化

        Args:
            actions: 行为列表
            learning_rate: 学习率
            reward_decay: 奖励衰减 gamma
            e_greedy:  贪婪度

        """
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, done, a_):
        """ 根据 已知值更新 Q table

        Args:
            s: 当前 state
            a: 当前 action
            r: 当前 reward
            s_: 下一个 state
            a_: 下一步动作

        Returns: 新 Q table

        """
        self.check_state_exist(s_)  # 检测 q_table 中是否存在 s_
        q_predict = self.q_table.loc[s, a]  # 采取当前行动后，预测得分
        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, a]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        update = self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += update  # 执行更新


import sys
from maze_env import Maze
import time


def maze_update(env, RL, episodes=100, learn=True):
    # 学习 100 回合
    for episode in range(episodes):
        # 初始化 state 的观测值
        observation = env.reset()

        # Sarsa 根据 state 观测选择行为
        action = RL.choose_action(str(observation))

        while True:
            # 更新可视化环境
            env.render()

            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)

            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            if learn:
                # RL 从这个序列 (state, action, reward, state_, action_) 中学习
                RL.learn(str(observation), action, reward, str(observation_), done, action_)

            # 将下一个 当成下一步的 state (observation) and action
            observation = observation_
            action = action_

            # 如果掉下地狱或者升上天堂, 这回合就结束了
            if done:  # 回合结束，跳出本轮循环
                break
        # time.sleep(5)
    if not learn:
        if reward == -1:
            print('测试失败')
        elif reward == 1:
            print('测试成功')
    else:
        print('学习结束')
    env.destroy()
    time.sleep(2)


def maze_test():
    # env = Maze()  # 生成环境获取参数
    # RL = QLearningTable(actions=list(range(env.n_actions)))  # 生成Qtable

    RL = SarsaTable(actions=list(range(4)))  # 生成Qtable

    for i in range(10):  # 测试10次，每次学10轮
        train_env = Maze('train')
        # train_env.after(100, lambda: maze_update(train_env, RL, 10, True))  # 学习10轮
        maze_update(train_env, RL, 10, True)  # 学习10轮
        print(i, 'q_table长度：', len(RL.q_table))
        test_env = Maze('test')
        start = time.time()
        # test_env.after(100, lambda: maze_update(test_env, RL, 1, False))  # 测试1轮
        maze_update(test_env, RL, 1, False)  # 测试1轮
        end = time.time()
        print(i, '测试时间：', end - start)
    # env.mainloop()

    # print('q_table索引', RL.q_table.index)
    # print('q_table列名', RL.q_table.columns)
    print(RL.q_table.head())


if __name__ == "__main__":
    maze_test()
