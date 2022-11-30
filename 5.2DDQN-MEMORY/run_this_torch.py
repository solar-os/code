from MyUser4 import BaseStation
from br_brain_torch import DQN
from br_brain_torch import Net
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

REWARD = []


def run():
    step = 0
    for epsilon in range(1000):
        observation = state[0]
        for i in range(16):
            action = RL.choose_action(observation, 0.9)  # 根据当前的状态值选择动作
            bs.offload.append(action)
            if i < 15:
                observation_ = state[i+1]
                reward = bs.action_Reward(i, action)
            else:
                observation_ = state[i]
                reward = bs.action_Reward(i, action)
            # print(observation, action, reward, observation_)
            REWARD.append(reward)
            RL.store_transition(observation, action, reward, observation_)
            if step > 100:
                RL.learn()
            observation = observation_
            step += 1


if __name__ == "__main__":
    actions = [0, 1]
    n_features = 2
    n_action = len(actions)
    state = [[12, 60],
             [16, 150],
             [300, 60],
             [21, 105],
             [380, 190],
             [16, 80],
             [140, 70],
             [20, 100],
             [12, 60],
             [16, 150],
             [300, 60],
             [21, 105],
             [380, 190],
             [16, 80],
             [140, 70],
             [20, 100]]
    task_size = [12, 16, 300, 21, 380, 16, 140, 20, 12, 16, 300, 21, 380, 16, 140, 20]  # 这里是输入任务的大小，单位bit，这里将输入全部去掉四个零
    taskList = [60, 150, 60, 105, 190, 80, 70, 100, 60, 150, 60, 105, 190, 80, 70, 100]  # 这里是每个任务需要的cpu的转数单位是Mcycle
    # 这里是任务依赖的前驱节点以及前驱节点输出数据大小
    outputRelationship = [[0, 0, 0, 0, 0, 0, 0, 0],
                          [1.2, 0, 0, 0, 0, 0, 0, 0],
                          [1.28, 0, 0, 0, 0, 0, 0, 0],
                          [1.2, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1.12, 0, 0, 0, 0, 0],
                          [0, 0, 1.28, 1.3, 0, 0, 0, 0],
                          [0, 0, 0, 1.44, 0, 0, 0, 0],
                          [0, 0.96, 0, 0, 0.96, 1.12, 1.04, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
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
    RL = DQN(n_features, n_action)
    run()

    # 保存模型
    RL.store_model()
    RL.plot_cost()
    RL.plot_reward(REWARD)

