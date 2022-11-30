import numpy as np
import math


#  代码优化 本次优化一下数字过大问题和代码冗余问题

class BaseStation:
    def __init__(self, f_local, f_mec, p_m, p_o, p_receive, p_send, p_AP, bandwidth, time_weight, power_weight,
                 taskList, outputRelationship, task_size):
        # 边缘服务器的cpu能力 上传功率 下载功率 任务队列 上传速率 等待时间 服务器的cpu能力 下载速率

        self.f_local = f_local  # 本地边缘设备的计算能力 设置成500MHZ
        self.f_mec = f_mec  # 边缘服务器的计算能力  设置成5000MHZ
        self.p_m = p_m  # 本地设备计算时的功率
        self.p_o = p_o  # 本地设备空闲时的功率
        self.p_receive = p_receive  # 本地设备接收计算结果时的功率
        self.p_send = p_send  # 本地设备发送中间结果时的功率
        self.p_AP = p_AP  # 基站的发射功率
        self.bandwidth = bandwidth  # 本地设备和基站之间的带宽
        self.time_weight = time_weight  # 时间权重
        self.power_weight = power_weight  # 功率权重
        self.taskList = taskList  # 任务列表
        self.outputRelationship = outputRelationship  # 任务之间的依赖关系
        self.task_size = task_size
        self.offload = []  # 卸载决策
        self.exe_time = []  # 卸载的总时间
        self.energy = []  # 能量消耗的总时间
        self.local = []  # 任务单独放在本地执行的时间 （不包括前驱节点的传输时间）
        self.edge = []  # 将任务发送到边缘设备所消耗的时间
        self.reward = []
        self.action_offload = []
        self.action_exe_time = []
        self.action_energy = []
        self.action_local = []
        self.action_edge = []
        self.action_reward = []

    # 用户上传数据到基站的数据传输速率
    def get_up_rate(self):
        #  print(2000000*math.log2(1+math.pow(10, 9)/math.pow(20,4)))
        return 8000000  # b/s  这里数据的大小用kb单位

    # 用户从基站下载数据的传输速率
    def get_down_rate(self):
        return 1150000  # 用户下载时的传输速率为 b/s

    # 判断任务是本地执行还是边缘执行,卸载决策存储在offload列表中，执行时间存储在exe_time列表中
    def make_decision(self, i):
        if i == 0:
            local_execute = self.taskList[i] / self.f_local  # 本地执行时间
            tran_time = self.task_size[i] / self.get_up_rate()  # 传输时间：任务传输到边缘的时间
            edge_execute = self.taskList[i] / self.f_mec  # 任务在边缘的执行时间
            edge_time = tran_time + edge_execute  # 任务传输到边缘和执行的时间之和
            self.local.append(local_execute)  # 将任务单独放在本地执行的时间放入local列表中
            self.edge.append(tran_time)  # 将任务单独放在边缘执行的时间放入edge列表中
            if local_execute < edge_time:
                self.offload.append(0)
                self.exe_time.append(local_execute)
                return [local_execute, edge_time]
            else:
                self.offload.append(1)
                self.exe_time.append(edge_time)
                return [local_execute, edge_time]
        else:
            temp_LocalTOEdge = {}
            temp_EdgeTOLocal = {}
            local_execute = self.taskList[i] / self.f_local  # 任务i在本地执行所需要的时间
            tran_time = self.task_size[i] / self.get_up_rate()  # 任务i上传到边缘服务器的时间
            edge_execute = self.taskList[i] / self.f_mec  # 任务i在边缘服务器上的执行时间
            edge_time = tran_time + edge_execute  # 任务i卸载到边缘服务器的总时间=传输时间+执行时间
            self.local.append(local_execute)  # 将任务单独放在本地执行的时间放入local列表中
            self.edge.append(tran_time)  # 将任务单独放在边缘执行的时间放入edge列表中
            getoutput = self.outputRelationship[i]  # 获取任务i依赖的前驱节点的中间数据列表
            getoutput = np.array(getoutput)  # 转换为数组
            getoutput_index = np.flatnonzero(getoutput)  # 获取非零值的索引值列表，非零值的索引也是其前驱节点的索引

            for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值
                if self.offload[getoutput_index[m]] == 0:  # 其前驱节点在本地执
                    print(getoutput[getoutput_index[m]])
                    # print('---索引----')
                    # print(getoutput_index[m])
                    comm_time_to_edge = getoutput[getoutput_index[m]] / self.get_up_rate()  # 计算中间结果传输到边缘所需要的时间
                    temp_LocalTOEdge['LocalToEdge' + str(m)] = comm_time_to_edge
                else:  # 其前驱节点在边缘执行
                    print(getoutput[getoutput_index[m]])
                    # print('---索引----')
                    # print(getoutput_index[m])
                    comm_time_to_local = getoutput[getoutput_index[m]] / self.get_down_rate()  # 中间结果传输到本地需要的时间
                    temp_EdgeTOLocal['EdgeTOLocal' + str(m)] = comm_time_to_local

            if temp_EdgeTOLocal:
                local_execute = local_execute + max(temp_EdgeTOLocal.values())  # 选取传输最大值
            if temp_LocalTOEdge:
                edge_time = edge_time + max(temp_LocalTOEdge.values())
            if local_execute < edge_time:
                self.offload.append(0)
                self.exe_time.append(local_execute)
            else:
                self.offload.append(1)
                self.exe_time.append(edge_time)
            return [local_execute, edge_time]

    # 通过读取卸载决策offload列表和执行时间列表exe_time，计算每个任务节点的能量消耗，这里的i为任务的索引号
    def energy_consumption(self, i):
        if i == 0:
            if self.offload[i] == 0:  # 任务i在本地执行
                energy_compute_local = self.local[i] * self.p_m  # 第一个节点在本地执行时的能量消耗  计算能耗
                self.energy.append(energy_compute_local)
                print('-----')
                print(energy_compute_local)
            else:  # 任务i在边缘执行
                energy_compute_edge = self.edge[i] * self.p_send  # 任务i在边缘执行时，本地设备需要考虑发射能耗，类似通信能耗
                self.energy.append(energy_compute_edge)
                print('-----')
                print(energy_compute_edge)
        else:
            if self.offload[i] == 0:  # 任务i在本地执行
                energy_compute_local = self.local[i] * self.p_m  # 任务i在本地执行，会消耗本地设备的计算能耗
                # 找到任务i在边缘中执行的前驱节点
                energy_commToLocal = 0
                getoutput = self.outputRelationship[i]  # 获取任务i依赖的前驱节点的中间数据列表
                getoutput = np.array(getoutput)  # 转换为数组
                getoutput_index = np.flatnonzero(getoutput)  # 获取非零值的索引值列表，非零值的索引也是其前驱节点的索引
                for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值
                    if self.offload[getoutput_index[m]] == 1:  # 其前驱节点在边缘执行
                        # print(getoutput[getoutput_index[m]])
                        # print('---索引----')
                        # print(getoutput_index[m])
                        comm_time_to_local = getoutput[getoutput_index[m]] / self.get_down_rate()  # 中间结果传输到本地需要的时间
                        energy_commToLocal = energy_commToLocal + self.p_receive * comm_time_to_local
                energy_local = energy_compute_local + energy_commToLocal
                self.energy.append(energy_local)
                print('------')
                print(energy_local)
            else:  # 任务i在边缘执行
                energy_compute_edge = self.edge[i] * self.p_send  # 任务在边缘执行，会消耗本地设备的发射能耗
                energy_commToEdge = 0
                getoutput = self.outputRelationship[i]  # 获取任务i依赖的前驱节点的中间数据列表
                getoutput = np.array(getoutput)  # 转换为数组
                getoutput_index = np.flatnonzero(getoutput)  # 获取非零值的索引值列表，非零值的索引也是其前驱节点的索引
                for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值
                    if self.offload[getoutput_index[m]] == 0:  # 其前驱节点在本地执行
                        comm_time_to_edge = getoutput[getoutput_index[m]] / self.get_up_rate()  # 计算中间结果传输到边缘所需要的时间
                        energy_commToEdge = energy_commToEdge + comm_time_to_edge * self.p_send
                energy_edge = energy_compute_edge + energy_commToEdge
                self.energy.append(energy_edge)
                print('-------')
                print(energy_edge)

    #  根据动作设置奖励值
    def action_Reward(self, i, a):  # i表示任务的索引号，a是所选的动作
        if i == 0:   # 表示是第一个任务
            if a == 0:   # 表示网络所选给任务做的决策是在本地执行，则下面就是计算在本地执行的reward值
                local_execute_time = self.taskList[i] / self.f_local   # 任务在本地执行所需要的时间
                local_compute_energy = local_execute_time * self.p_m           # 任务i在本地执行所消耗的计算能耗
                action_reward = self.time_weight * local_execute_time + self.power_weight * local_compute_energy
                action_reward = round(action_reward, 4)
                return -action_reward
            else:  # 表示第一个任务网络做的决策是卸载到边缘服务器端执行
                tran_to_edge_time = self.task_size[i] / self.get_up_rate()   # 计算任务上传到边缘所消耗的时间
                edge_execute_time = self.taskList[i] / self.f_mec            # 计算任务在边缘执行所消耗的时间
                edge_offload_time = tran_to_edge_time + edge_execute_time             # 计算卸载到边缘所花费的总时间
                tran_to_edge_energy = tran_to_edge_time * self.p_send        # 计算任务卸载到边缘所花费的能耗
                action_reward = self.time_weight * edge_offload_time + self.power_weight * tran_to_edge_energy
                action_reward = round(action_reward, 4)                 # 奖励值保留到小数点后4位
                return -action_reward
        else:    # 表示任务不是第一个任务
            getoutput = self.outputRelationship[i]  # 获取任务i依赖的前驱节点的中间数据列表
            getoutput = np.array(getoutput)  # 转换为数组
            getoutput_index = np.flatnonzero(getoutput)  # 获取非零值的索引值列表，非零值的索引也是其前驱节点的索引
            temp_commTime_EdgeToLocal = {}
            temp_commTime_LocalToEdge = {}
            if a == 0:   # 表示任务i在本地执行
                local_execute_time = self.taskList[i] / self.f_local   # 任务在本地执行所需要的时间
                local_compute_energy = local_execute_time * self.p_m   # 任务在本地执行所消耗的能耗
                comm_Edge_To_Local_energy = 0
                # 下面计算任务和其前驱节点在的通信时间和通信能耗，由于任务i在本地执行，只有其前驱节点在边缘执行时才会产生通信时间和通信能耗
                for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值 getoutput_index[m]位前驱节点的索引值
                    if self.offload[getoutput_index[m]] == 1:  # 其前驱节点在边缘执行
                        # 起算其前驱节点的输出传输到本地所需要的时间
                        commTime_EdgeToLocal = getoutput[getoutput_index[m]] / self.get_down_rate()
                        # 计算任务i和其前驱节点通信所消耗的能耗
                        comm_Edge_To_Local_energy = comm_Edge_To_Local_energy + commTime_EdgeToLocal * self.p_receive
                        temp_commTime_EdgeToLocal['EdgeToLocal'+str(m)] = commTime_EdgeToLocal
                if temp_commTime_EdgeToLocal:     # 如果任务i存在前驱节点在边缘执行
                    local_sum_time = local_execute_time + max(temp_commTime_EdgeToLocal.values())  # 任务i在本地的总执行时间为：本地执行时间+中间结果的传输时间
                else:
                    local_sum_time = local_execute_time
                # 任务i在本地执行所消耗的能耗为：本地计算能耗 + 和前驱节点通信能耗
                local_sum_energy = local_compute_energy + comm_Edge_To_Local_energy
                action_reward = self.time_weight * local_sum_time + self.power_weight * local_sum_energy
                action_reward = round(action_reward, 4)
                return -action_reward
            else:     # 任务i在边缘执行
                tran_to_edge_time = self.task_size[i] / self.get_up_rate()  # 任务i传输到边缘所消耗的时间
                edge_execute = self.taskList[i] / self.f_mec  # 任务i在边缘执行的时间
                edge_offload_time = tran_to_edge_time + edge_execute  # 任务i在卸载到边缘的总时间：传输时间+执行时间
                tran_to_edge_energy = tran_to_edge_time * self.p_send  # 任务i发送到边缘所消耗的发射能耗
                comm_Local_To_Edge_energy = 0
                for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值 getoutput_index[m]位前驱节点的索引值
                    if self.offload[getoutput_index[m]] == 0:  # 其前驱节点在本地执行
                        # 其前驱节点的结果从本地上传到边缘所消耗的时间
                        commTime_LocalToEdge = getoutput[getoutput_index[m]] / self.get_up_rate()
                        comm_Local_To_Edge_energy = comm_Local_To_Edge_energy + commTime_LocalToEdge * self.p_send
                        temp_commTime_LocalToEdge['LocalToEdge'+str(m)] = commTime_LocalToEdge
                if temp_commTime_LocalToEdge:
                    # 任务i在边缘的总时间为：任务i卸载到边缘的时间 + 任务i与其前驱节点的通信时间
                    edge_sum_time = edge_offload_time + max(temp_commTime_LocalToEdge.values())
                else:
                    edge_sum_time = edge_offload_time
                # 任务i卸载到边缘消耗本地设备的能耗为：将任务传输到边缘的传输能耗+其在本地执行的前驱节点的中间结果传输到边缘的传输能耗
                edge_sum_energy = tran_to_edge_energy + comm_Local_To_Edge_energy
                action_reward = self.time_weight * edge_sum_time + self.power_weight * edge_sum_energy
                action_reward = round(action_reward, 4)    # 将奖励值保留小数点后四位
                return -action_reward

    # 计算任务的奖励值
    def Reward(self, i):
        reward = self.time_weight * self.exe_time[i] + self.power_weight * self.energy[i]
        self.reward.append(reward)
        return reward


if __name__ == '__main__':
    task_size = [120000, 160000, 3000000, 210000, 3800000, 160000, 1400000, 200000]  # 这里是输入任务的大小，单位bit
    taskList = [60, 150, 60, 105, 190, 80, 70, 100]  # 这里是每个任务需要的cpu的转数单位是Mcycle
    # 这里是任务依赖的前驱节点以及前驱节点输出数据大小
    outputRelationship = [[0, 0, 0, 0, 0, 0, 0, 0],
                          [12000, 0, 0, 0, 0, 0, 0, 0],
                          [12800, 0, 0, 0, 0, 0, 0, 0],
                          [12000, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 11200, 0, 0, 0, 0, 0],
                          [0, 0, 12800, 13000, 0, 0, 0, 0],
                          [0, 0, 0, 14400, 0, 0, 0, 0],
                          [0, 9600, 0, 0, 9600, 11200, 10400, 0]]
    taskList2 = [60, 80, 150, 100]
    outputRelationship2 = [[0, 0, 0, 0],
                           [8, 0, 0, 0],
                           [12, 0, 0, 0],
                           [0, 12, 16, 0]]
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
    for i in range(len(taskList)):
        print(i)
        bs.make_decision(i)
        bs.energy_consumption(i)
        bs.Reward(i)

    print(bs.offload)
    exe_time = np.array(bs.exe_time)
    exe_time = np.round(exe_time, 4)  # 将执行时间保留为小数点后四位
    print(exe_time)
    energy = np.array(bs.energy)
    energy = np.round(energy, 4)  # 任务的能耗保留为小数点后四位
    print(energy)
    reward = np.array(bs.reward)
    reward = np.round(reward, 4)  # 将任务的奖励值保留为小数点后四位
    print(reward)
    print(bs.action_Reward(7, 1))
