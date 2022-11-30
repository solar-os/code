from collections import defaultdict
from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MyUser4 import BaseStation

actions = [0, 1]
state = 8
alpha = 0.1
epsilon = 0.1
discount_factor = 0.5
max_episode = 600
taskList = [[800, 1150, 12, 60],
            [800, 1150, 16, 150],
            [800, 1150, 300, 60],
            [800, 1150, 21, 105],
            [800, 1150, 380, 190],
            [800, 1150, 16, 80],
            [800, 1150, 140, 70],
            [800, 1150, 20, 100]]
'''
taskList = [[2, 1, 3],
            [3, 4, 5],
            [1, 50, 10],
            [2, 1, 10],
            [5, 1, 100]]  # 用户任务上产大小为2，任务结果大小为1，所需要的cpu周期为3
'''
# print(taskList[0])
state = taskList
state_num = len(state)
# bs = BaseStation(10, 5, 2, 1, 4, 4, 6, 100, 5, 0.5, 0.5, taskList)
task_size = [12, 16, 300, 21, 380, 16, 140, 20]  # 这里是输入任务的大小，单位bit，这里将输入全部去掉四个零
taskList = [60, 150, 60, 105, 190, 80, 70, 100]  # 这里是每个任务需要的cpu的转数单位是Mcycle
outputRelationship = [[0, 0, 0, 0, 0, 0, 0, 0],
                      [1.2, 0, 0, 0, 0, 0, 0, 0],
                      [1.28, 0, 0, 0, 0, 0, 0, 0],
                      [1.2, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1.12, 0, 0, 0, 0, 0],
                      [0, 0, 1.28, 1.3, 0, 0, 0, 0],
                      [0, 0, 0, 1.44, 0, 0, 0, 0],
                      [0, 0.96, 0, 0, 0.96, 1.12, 1.04, 0]]

bs = BaseStation(
    f_local=500,
    f_mec=5000,
    p_m=0.5,
    p_o=0.01,
    p_receive=0.05,
    p_send=0.1,
    p_AP=1,
    bandwidth=2,
    time_weight=0.5,
    power_weight=0.5,
    taskList=taskList,
    outputRelationship=outputRelationship,
    task_size=task_size
)


# task = taskList[0]
# print("----")
# print(task[0])
# 创建Q表
def buid_Qtable(state_num, actions):
    Qtable = pd.DataFrame(np.zeros((state_num, len(actions))), columns=actions)
    return Qtable


# 根据Q表和状态值选择动作
def ChooseAction(state, Qtable):
    if np.random.rand() > epsilon:
        action_name = np.argmax(Qtable.iloc[state, :])
    else:
        action_name = np.random.choice(actions)
    return action_name


def rl():
    Qtable = buid_Qtable(state_num, actions)  # 创建Q表
    for episode in range(max_episode):
        # m, _ = np.array(taskList).shape
        cur_state = taskList[0]  # 随机初始化一个状态为0
        bs.offload = []
        for i in range(8):
            # print(i)
            cur_action = ChooseAction(i, Qtable)  # 根据Q表选择动作
            bs.offload.append(cur_action)
            if i < 7:
                new_state = taskList[i + 1]
                reward = bs.Independent_tasks_action_reward(i, cur_action)  # 将任务当成我们的状态空间，则任务应当与奖励值绑定，所以这里传过去应当有任务
                Q_predict = Qtable.loc[i, cur_action]
                Q_target = reward + discount_factor * Qtable.iloc[i, :].max()
                Qtable.loc[i, cur_action] += alpha * (Q_target - Q_predict)
                cur_state = new_state
            else:
                reward = bs.Independent_tasks_action_reward(i, cur_action)  # 将任务当成我们的状态空间，则任务应当与奖励值绑定，所以这里传过去应当有任务
                Q_predict = Qtable.loc[i, cur_action]
                Q_target = reward + discount_factor * Qtable.iloc[i, :].max()
                Qtable.loc[i, cur_action] += alpha * (Q_target - Q_predict)
    return Qtable


q_tabel = rl()
print(q_tabel)
