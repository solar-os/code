import torch
from br_brain_torch import Net
from MyUser4 import BaseStation

m_state_dict = torch.load('eval_net_DQN.pt')
new_m = Net(2, 2)
new_m.load_state_dict(m_state_dict)
DATA = [[12, 60],
        [16, 150],
        [300, 60],
        [21, 105],
        [380, 190],
        [16, 80],
        [140, 70],
        [20, 100]]
for j in range(len(DATA)):
    print(j)
    d = DATA[j]
    d = torch.FloatTensor(d)
    predict = new_m(d)
    print(predict)
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
taskList2 = [60, 80, 150, 100]
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

print(predict)

