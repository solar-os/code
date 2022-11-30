from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import numpy as np
# from MyUser3 import BaseStation  # 依赖人物的时候用
from MyUser4 import BaseStation
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

MEMORY_SIZE = 100

RL_natural = DQNPrioritizedReplay(n_actions=2, n_features=2, double_q=False, memory_size=MEMORY_SIZE,
                                  e_greedy_increment=0.00005, prioritized=False)
RL_prio = DQNPrioritizedReplay(n_actions=2, n_features=2, double_q=True, memory_size=MEMORY_SIZE,
                               e_greedy_increment=0.00005, prioritized=True)


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(800):     # 500的时候训练结果已经很好了
        observation = STATE[0]
        bs.offload = []
        for i in range(16):
            action = RL.choose_action(observation)
            bs.offload.append(action)  # 有依赖关系任务的时候用
            if i < 15:
                observation_ = STATE[i + 1]
                # print(observation)
                # 根据所选择的action计算过奖励值
                reward = bs.action_Reward(i, action)    # 有依赖关系的奖励值
                # reward = bs.Independent_tasks_action_reward(i, action)
            else:
                observation_ = STATE[i]
                # print(observation)
                # reward = bs.Independent_tasks_action_reward(i, action)
                reward = bs.action_Reward(i, action)  # 有依赖关系的奖励值
            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if i == 15:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break
            observation = observation_
            total_steps += 1
        print("steps for {}th episode: {}".format(i_episode, total_steps))
    return np.vstack((episodes, steps)), RL.q


actions = [0, 1]
n_features = 2
n_action = len(actions)
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
outputRelationship2 = [[0, 0, 0, 0, 0, 0, 0, 0],
                       [0.8, 0, 0, 0, 0, 0, 0, 0],
                       [1.28, 0, 0, 0, 0, 0, 0, 0],
                       [1.12, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1.28, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1.04, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1.44, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1.6, 1.2, 1.6, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0.8, 0, 0, 0, 0, 0, 0, 0],
                       [1.28, 0, 0, 0, 0, 0, 0, 0],
                       [1.12, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1.28, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1.04, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1.44, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1.6, 1.2, 1.6, 0]
                       ]
outputRelationship3 = [[0, 0, 0, 0, 0, 0, 0, 0],
                       [1.28, 0, 0, 0, 0, 0, 0, 0],
                       [0.96, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1.12, 0, 0, 0, 0, 0, 0],
                       [0, 1.44, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1.04, 0, 0, 0, 0, 0],
                       [0, 0, 1.2, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0.8, 1.6, 0.8, 1.44, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [1.28, 0, 0, 0, 0, 0, 0, 0],
                       [0.96, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1.12, 0, 0, 0, 0, 0, 0],
                       [0, 1.44, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1.04, 0, 0, 0, 0, 0],
                       [0, 0, 1.2, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0.8, 1.6, 0.8, 1.44, 0]
                       ]

STATE = [[12, 60],
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
         [20, 100]
         ]
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
    outputRelationship=outputRelationship3,
    task_size=task_size
)

his_natural, DQN = train(RL_natural)
his_prio, DDQN = train(RL_prio)

RL_prio.store_model()
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[0, :], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[0, :], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()
RL_prio.plot_cost()
plt.plot(np.array(DQN), c='r', label='natural')
plt.plot(np.array(DDQN), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()
