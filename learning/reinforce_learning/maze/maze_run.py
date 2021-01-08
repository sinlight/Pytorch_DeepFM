"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import sys

from maze_env import Maze
from RL_brain import QLearningTable
import time


def update():
    # 学习 100 回合
    for episode in range(100):
        # 初始化 state 的观测值
        observation = env.reset()

        while True:
            # 更新可视化环境
            env.render()

            # RL 大脑根据 state 的观测值挑选 action
            action = RL.choose_action(str(observation))

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)

            # RL 从这个序列 (state, action, reward, state_) 中学习
            RL.learn(str(observation), action, reward, str(observation_))

            # 将下一个 state 的值传到下一次循环
            observation = observation_

            # 如果掉下地狱或者升上天堂, 这回合就结束了
            if done:
                break

        # time.sleep(5)

    print('q_table长度', len(RL.q_table))
    print('q_table索引', RL.q_table.index)
    print('q_table列名', RL.q_table.columns)
    # print(RL.q_table.head())

    # end of game
    print('game over')
    env.destroy()
    time.sleep(2)


if __name__ == "__main__":
    env = Maze()  # 生成环境
    RL = QLearningTable(actions=list(range(env.n_actions)))  # 生成Qtable

    env.after(100, update)
    env.mainloop()
