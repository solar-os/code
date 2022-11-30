import numpy as np
import math


#  代码优化 本次优化一下数字过大问题和代码冗余问题
# 2022-1-10 用户代码本次优化用户设备的cpu的频率选择问题

class BaseStation:
    def __init__(self, f_local, f_local_two, f_mec, p_m, p_o, p_receive, p_send, p_send_two, p_AP, bandwidth,
                 time_weight,
                 power_weight, taskList, outputRelationship, task_size):
        # 边缘服务器的cpu能力 上传功率 下载功率 任务队列 上传速率 等待时间 服务器的cpu能力 下载速率
        self.f_local = f_local  # 本地边缘设备的计算能力 设置成500MHZ
        self.f_local_two = f_local_two  # 本地设备的第二个CPU频率，设置为1500MHZ
        self.f_mec = f_mec  # 边缘服务器的计算能力  设置成5000MHZ
        self.p_m = p_m  # 本地设备计算时的功率
        self.p_o = p_o  # 本地设备空闲时的功率
        self.p_receive = p_receive  # 本地设备接收计算结果时的功率
        self.p_send = p_send  # 本地设备发送中间结果时的功率
        self.p_send_two = p_send_two  # 本地设备发送中间结果时的功率
        self.p_AP = p_AP  # 基站的发射功率
        self.bandwidth = bandwidth  # 本地设备和基站之间的带宽
        self.time_weight = time_weight  # 时间权重
        self.power_weight = power_weight  # 功率权重
        self.taskList = taskList  # 任务列表
        self.outputRelationship = outputRelationship  # 任务之间的依赖关系
        self.task_size = task_size
        # self.offload = [1, 0, 1, 0, 1, 0, 1, 1]  # 卸载决策
        # self.offload = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
        self.offload = []
        self.offload_two = []
        self.exe_time = []  # 有依赖关系的卸载的总时间
        self.exe_energy = []
        self.exe_time_v1 = []
        self.exe_energy_v1 = []
        self.energy = []  # 有依赖关系的能量消耗的总时间
        self.local = []  # 任务单独放在本地执行的时间 （不包括前驱节点的传输时间）
        self.edge = []  # 将任务发送到边缘设备所消耗的时间
        self.reward = []
        self.action_offload = []
        self.action_exe_time = []
        self.action_energy = []
        self.action_local = []
        self.action_edge = []
        self.Action_Reward = [0.0143, 0.026, 0.0906, 0.0249, 0.285, 0.0199, 0.1057, 0.0245, 0.0143, 0.026, 0.0906,
                              0.0249, 0.285, 0.0199, 0.1057, 0.0245]  # 通用任务图的奖励值
        # self.Action_Reward = [0.0143, 0.026, 0.0906, 0.0249, 0.285, 0.0197, 0.1057, 0.025]    # 普通链型任务图
        # self.Action_Reward = [0.0143, 0.026, 0.0904, 0.0249, 0.2857, 0.0197, 0.105, 0.0249]      # 树形结构
        self.Action_Reward_TWO = [0.0143, 0.026, 0.09, 0.0249, 0.285, 0.019, 0.105, 0.0238]  # 独立任务的奖励值
        self.offload_independence = []
        self.exe_time_Independent = []  # 记录独立任务的执行时间
        self.energy_Independent = []  # 记录独立任务的能量
        self.sum_time = []
        self.sum_energy = []

    # 用户上传数据到基站的数据传输速率
    def get_up_rate(self):
        #  print(2000000*math.log2(1+math.pow(10, 9)/math.pow(20,4)))
        return 500  # b/s   这里将去掉了四个零

    def get_up_rate_two(self):
        #  print(2000000*math.log2(1+math.pow(10, 9)/math.pow(20,4)))
        return 1000  # b/s   这里将去掉了四个零

    # 用户从基站下载数据的传输速率
    def get_down_rate(self):
        return 714  # 用户下载时的传输速率为 b/s

    # -------------------------------
    # 通用任务图的决策和根据动作生成奖励值
    # 判断任务是本地执行还是边缘执行,卸载决策存储在offload列表中，执行时间存储在exe_time列表中
    def make_decision(self, i):
        if i == 0:
            local_execute = self.taskList[i] / self.f_local  # 本地执行时间
            tran_time = self.task_size[i] / self.get_up_rate()  # 传输时间：任务传输到边缘的时间
            edge_execute = self.taskList[i] / self.f_mec  # 任务在边缘的执行时间
            edge_time = tran_time + edge_execute  # 任务传输到边缘和执行的时间之和
            self.local.append(local_execute)  # 将任务单独放在本地执行的时间放入local列表中
            self.edge.append(tran_time)  # 将任务单独放在边缘执行的时间放入edge列表中
            if local_execute < edge_time:  # 比较本地执行时间和下载到边缘执行的时间
                self.offload.append(0)  # 卸载策略
                self.exe_time.append(local_execute)  # 该卸载策略下的执行时间
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
            # 如果该任务不是第一个任务则需要考虑其前驱的依赖任务，分别判断前驱任务是在本地执行还是边缘执行
            for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值
                if self.offload[getoutput_index[m]] == 0:  # 其前驱节点在本地执
                    # print(getoutput[getoutput_index[m]])
                    # print('---索引----')
                    # print(getoutput_index[m])
                    comm_time_to_edge = getoutput[getoutput_index[m]] / self.get_up_rate()  # 计算中间结果传输到边缘所需要的时间
                    temp_LocalTOEdge['LocalToEdge' + str(m)] = comm_time_to_edge
                else:  # 其前驱节点在边缘执行
                    # print(getoutput[getoutput_index[m]])
                    # print('---索引----')
                    # print(getoutput_index[m])
                    comm_time_to_local = getoutput[getoutput_index[m]] / self.get_down_rate()  # 中间结果传输到本地需要的时间
                    temp_EdgeTOLocal['EdgeTOLocal' + str(m)] = comm_time_to_local

            if temp_EdgeTOLocal:
                local_execute = local_execute + max(temp_EdgeTOLocal.values())  # 这里依赖任务可能不止一个，我们取最大值
            if temp_LocalTOEdge:
                edge_time = edge_time + max(temp_LocalTOEdge.values())
            if local_execute < edge_time:
                self.offload.append(0)
                self.exe_time.append(local_execute)
            else:
                self.offload.append(1)
                self.exe_time.append(edge_time)
            return [local_execute, edge_time]

    # 这里的卸载决策仅考虑了时间和能耗的加权和
    def make_decision_two(self, i):
        if i == 0:  # 这里判断第一个任务是在本地执行还是在边缘执行
            local_execute = self.taskList[i] / self.f_local  # 本地执行时间
            local_compute_energy = math.pow(10, -27) * math.pow(self.f_local * 1000000, 2) * self.taskList[
                    i] * 1000000  # 任务i在本地执行所消耗的计算能耗
            tran_time = self.task_size[i] / self.get_up_rate()  # 传输时间：任务传输到边缘的时间
            edge_execute = self.taskList[i] / self.f_mec  # 任务在边缘的执行时间
            tran_to_edge_energy = tran_time * self.p_send  # 计算任务卸载到边缘所花费的能耗
            edge_time = tran_time + edge_execute  # 任务传输到边缘和执行的时间之和
            TT_local = self.time_weight * local_execute + self.power_weight * local_compute_energy
            TT_edge = self.time_weight * edge_time + self.power_weight * tran_to_edge_energy
            if TT_local < TT_edge:
                self.offload.append(0)
                self.exe_time_v1.append(local_execute)
                self.exe_energy_v1.append(local_compute_energy)
            else:
                self.offload.append(1)
                self.exe_time_v1.append(edge_time)
                self.exe_energy_v1.append(tran_to_edge_energy)
            return [TT_local, TT_edge]
        else:
            temp_LocalTOEdge = {}
            temp_EdgeTOLocal = {}
            local_execute = self.taskList[i] / self.f_local  # 任务i在本地执行所需要的时间
            local_compute_energy = math.pow(10, -27) * math.pow(self.f_local * 1000000, 2) * self.taskList[
                    i] * 1000000  # 任务i在本地执行所消耗的计算能耗
            tran_time = self.task_size[i] / self.get_up_rate()  # 任务i上传到边缘服务器的时间
            tran_to_edge_energy = tran_time * self.p_send  # 计算任务卸载到边缘所花费的能耗
            edge_execute = self.taskList[i] / self.f_mec  # 任务i在边缘服务器上的执行时间
            edge_time = tran_time + edge_execute  # 任务i卸载到边缘服务器的总时间=传输时间+执行时间
            self.local.append(local_execute)  # 将任务单独放在本地执行的时间放入local列表中
            self.edge.append(tran_time)  # 将任务单独放在边缘执行的时间放入edge列表中
            getoutput = self.outputRelationship[i]  # 获取任务i依赖的前驱节点的中间数据列表
            getoutput = np.array(getoutput)  # 转换为数组
            getoutput_index = np.flatnonzero(getoutput)  # 获取非零值的索引值列表，非零值的索引也是其前驱节点的索引
            comm_time_to_edge_energy = 0
            comm_time_to_local_energy = 0
            for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值
                if self.offload[getoutput_index[m]] == 0:  # 其前驱节点在本地执
                    # print(getoutput[getoutput_index[m]])
                    # print('---索引----')
                    # print(getoutput_index[m])
                    comm_time_to_edge = getoutput[getoutput_index[m]] / self.get_up_rate()  # 计算中间结果传输到边缘所需要的时间
                    comm_time_to_edge_energy += comm_time_to_edge * self.p_send
                    temp_LocalTOEdge['LocalToEdge' + str(m)] = comm_time_to_edge
                else:  # 其前驱节点在边缘执行
                    # print(getoutput[getoutput_index[m]])
                    # print('---索引----')
                    # print(getoutput_index[m])
                    comm_time_to_local = getoutput[getoutput_index[m]] / self.get_down_rate()  # 中间结果传输到本地需要的时间
                    comm_time_to_local_energy += comm_time_to_local * self.p_receive
                    temp_EdgeTOLocal['EdgeTOLocal' + str(m)] = comm_time_to_local

            if temp_EdgeTOLocal:
                local_execute = local_execute + max(temp_EdgeTOLocal.values())  # 选取传输最大值
            if temp_LocalTOEdge:
                edge_time = edge_time + max(temp_LocalTOEdge.values())
            TT_local = local_execute * self.time_weight + (
                    local_compute_energy + comm_time_to_local_energy) * self.power_weight
            TT_edge = edge_time * self.time_weight + (
                    tran_to_edge_energy + comm_time_to_edge_energy) * self.power_weight
            if TT_local < TT_edge:
                self.offload.append(0)
                self.exe_time_v1.append(local_execute)  # 记录任务该卸载策略下执行的时间
                self.exe_energy_v1.append(local_compute_energy + comm_time_to_local_energy)  # 记录任务的卸载策略下的能耗
            else:
                self.offload.append(1)
                self.exe_time_v1.append(edge_time)
                self.exe_energy_v1.append(tran_to_edge_energy + comm_time_to_edge_energy)
            return [TT_local, TT_edge]

    # 通过读取卸载决策offload列表和执行时间列表exe_time，计算每个任务节点的能量消耗，这里的i为任务的索引号
    def energy_consumption(self, i):
        if i == 0:
            if self.offload[i] == 0:  # 任务i在本地执行
                energy_compute_local = self.local[i] * self.p_m  # 第一个节点在本地执行时的能量消耗  计算能耗
                self.energy.append(energy_compute_local)
            else:  # 任务i在边缘执行
                energy_compute_edge = self.edge[i] * self.p_send  # 任务i在边缘执行时，本地设备需要考虑发射能耗，类似通信能耗
                self.energy.append(energy_compute_edge)
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

    #  根据动作设置奖励值, 如果所选的动作是正确的，则奖励值设置为1，所选的动作是错误的，奖励值设置为0
    def action_Reward(self, i, a):  # i表示任务的索引号，a是所选的动作
        if i == 0:  # 表示是第一个任务
            if a == 0:  # 表示网络所选给任务做的决策是在本地执行，则下面就是计算在本地执行的reward值
                local_execute_one = self.taskList[i] / self.f_local  # 本地执行时间 f_local=500MHZ
                local_compute_energy_one = math.pow(10, -27) * math.pow(self.f_local * 1000000, 2) * self.taskList[
                    i] * 1000000  # 任务i在本地执行所消耗的计算能耗
                local_execute_two = self.taskList[i] / self.f_local_two
                local_compute_energy_two = math.pow(10, -27) * math.pow(self.f_local_two * 1000000, 2) * self.taskList[
                    i] * 1000000
                TT_local_one = self.time_weight * local_execute_one + self.power_weight * local_compute_energy_one
                TT_local_two = self.time_weight * local_execute_two + self.power_weight * local_compute_energy_two
                if TT_local_one < TT_local_two:
                    action_reward = TT_local_one
                    flag = 1  # flag是一个标记，1表示500MHZ的cpu频率下的时间和能耗加权和小，否则就是1500MHZ更好点
                else:
                    action_reward = TT_local_two
                    flag = 2
                action_reward = round(action_reward, 4)
                if flag == 1:
                    print("在本地执行，所选的cpu的频率为500MHZ,时间和能耗分别为")
                    print(local_execute_one)
                    print(local_compute_energy_one)
                else:
                    print("在本地执行，所选的cpu的频率为1500MHZ,时间和能耗分别为")
                    print(local_execute_two)
                    print(local_compute_energy_two)

            else:  # 表示第一个任务网络做的决策是卸载到边缘服务器端执行
                tran_time_one = self.task_size[i] / self.get_up_rate()  # 传输时间：任务传输到边缘的时间 用的是500的发送功率
                tran_time_two = self.task_size[i] / self.get_up_rate_two()  # 传输时间：任务传输到边缘的时间 用的是500的发送功率
                edge_execute = self.taskList[i] / self.f_mec  # 任务在边缘的执行时间
                tran_to_edge_energy_one = tran_time_one * self.p_send  # 计算任务卸载到边缘所花费的能耗
                tran_to_edge_energy_two = tran_time_two * self.p_send_two  # 计算任务卸载到边缘所花费的能耗 用的是第二个等级的功率
                edge_time_one = tran_time_one + edge_execute  # 任务传输到边缘和执行的时间之和
                edge_time_two = tran_time_two + edge_execute  # 任务传输到边缘和执行的时间之和
                TT_edge_one = self.time_weight * edge_time_one + self.power_weight * tran_to_edge_energy_one
                TT_edge_two = self.time_weight * edge_time_two + self.power_weight * tran_to_edge_energy_two
                if TT_edge_one < TT_edge_two:
                    action_reward = TT_edge_one
                    edge_flag = 1
                else:
                    action_reward = TT_edge_two
                    edge_flag = 2
                action_reward = round(action_reward, 4)  # 奖励值保留到小数点后4位
                if edge_flag == 1:
                    print("在边缘执行，所选的发射功率为500w,时间和能量分别为")
                    print(edge_time_one)
                    print(tran_to_edge_energy_one)
                else:
                    print("在边缘执行，所选的发射功率为1000w,时间和能量分别为")
                    print(edge_time_two)
                    print(tran_to_edge_energy_two)

        else:  # 表示任务不是第一个任务
            getoutput = self.outputRelationship[i]  # 获取任务i依赖的前驱节点的中间数据列表

            getoutput = np.array(getoutput)  # 转换为数组

            getoutput_index = np.flatnonzero(getoutput)  # 获取非零值的索引值列表，非零值的索引也是其前驱节点的索引
            temp_commTime_EdgeToLocal = {}
            temp_commTime_LocalToEdge = {}
            if a == 0:  # 表示任务i在本地执行
                local_execute_one = self.taskList[i] / self.f_local  # 本地执行时间 f_local=500MHZ
                local_compute_energy_one = math.pow(10, -27) * math.pow(self.f_local * 1000000, 2) * self.taskList[
                    i] * 1000000  # 任务i在本地执行所消耗的计算能耗
                local_execute_two = self.taskList[i] / self.f_local_two
                local_compute_energy_two = math.pow(10, -27) * math.pow(self.f_local_two * 1000000, 2) * self.taskList[
                    i] * 1000000
                comm_Edge_To_Local_energy = 0
                # 下面计算任务和其前驱节点在的通信时间和通信能耗，由于任务i在本地执行，只有其前驱节点在边缘执行时才会产生通信时间和通信能耗
                for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值 getoutput_index[m]位前驱节点的索引值
                    if self.offload[getoutput_index[m]] == 1:  # 其前驱节点在边缘执行
                        # 起算其前驱节点的输出传输到本地所需要的时间
                        commTime_EdgeToLocal = getoutput[getoutput_index[m]] / self.get_down_rate()
                        # 计算任务i和其前驱节点通信所消耗的能耗
                        comm_Edge_To_Local_energy = comm_Edge_To_Local_energy + commTime_EdgeToLocal * self.p_receive
                        temp_commTime_EdgeToLocal['EdgeToLocal' + str(m)] = commTime_EdgeToLocal
                if temp_commTime_EdgeToLocal:  # 如果任务i存在前驱节点在边缘执行
                    temp_flag = 1
                    TT_local_one_temp = local_execute_one + max(temp_commTime_EdgeToLocal.values())  # 选取传输最大值  本地执行的时间
                    TT_local_two_temp = local_execute_two + max(temp_commTime_EdgeToLocal.values())  # 选取传输最大值
                    TT_local_one = TT_local_one_temp * self.time_weight + (
                            local_compute_energy_one + comm_Edge_To_Local_energy) * self.power_weight
                    TT_local_two = TT_local_two_temp * self.time_weight + (
                            local_compute_energy_two + comm_Edge_To_Local_energy) * self.power_weight
                    if TT_local_one < TT_local_two:
                        local_sum_time = TT_local_one
                        local_sum_energy = local_compute_energy_one + comm_Edge_To_Local_energy
                        flag = 1
                    else:
                        local_sum_time = TT_local_two
                        local_sum_energy = local_compute_energy_two + comm_Edge_To_Local_energy
                        flag = 2

                else:
                    temp_flag = 2
                    TT_local_one_temp = local_execute_one  # 选取传输最大值  本地执行的时间
                    TT_local_two_temp = local_execute_two  # 选取传输最大值
                    TT_local_one = TT_local_one_temp * self.time_weight + (
                        local_compute_energy_one) * self.power_weight
                    TT_local_two = TT_local_two_temp * self.time_weight + (
                        local_compute_energy_two) * self.power_weight
                    if TT_local_one < TT_local_two:
                        flag = 1
                        local_sum_time = TT_local_one
                        local_sum_energy = local_compute_energy_one
                    else:
                        flag = 2
                        local_sum_time = TT_local_two
                        local_sum_energy = local_compute_energy_two
                        # 任务i在本地执行所消耗的能耗为：本地计算能耗 + 和前驱节点通信能耗
                action_reward = self.time_weight * local_sum_time + self.power_weight * local_sum_energy
                action_reward = round(action_reward, 4)
                if temp_flag == 1:
                    if flag == 1:
                        print("所选的cpu的频率为500MHZ，有前驱节点在边缘执行,在本地执行的时间和能耗分别为")
                        print(TT_local_one)
                        print(local_sum_energy)
                    else:
                        print("所选的cpu的频率为1500MHZ，有前驱节点在边缘执行,在本地执行的时间和能耗分别为")
                        print(TT_local_two)
                        print(local_sum_energy)
                else:
                    if flag == 1:
                        print("所选的cpu的频率为500MHZ，无前驱节点在边缘执行,在本地执行的时间和能耗分别为")
                        print(TT_local_one)
                        print(local_sum_energy)
                    else:
                        print("所选的cpu的频率为1500MHZ，无前驱节点在边缘执行,在本地执行的时间和能耗分别为")
                        print(TT_local_two)
                        print(local_sum_energy)

            else:  # 任务i在边缘执行
                tran_time_one = self.task_size[i] / self.get_up_rate()  # 传输时间：任务传输到边缘的时间 用的是500的发送功率
                tran_time_two = self.task_size[i] / self.get_up_rate_two()  # 传输时间：任务传输到边缘的时间 用的是500的发送功率
                edge_execute = self.taskList[i] / self.f_mec  # 任务在边缘的执行时间
                tran_to_edge_energy_one = tran_time_one * self.p_send  # 计算任务卸载到边缘所花费的能耗
                tran_to_edge_energy_two = tran_time_two * self.p_send_two  # 计算任务卸载到边缘所花费的能耗 用的是第二个等级的功率
                edge_time_one = tran_time_one + edge_execute  # 任务传输到边缘和执行的时间之和
                edge_time_two = tran_time_two + edge_execute  # 任务传输到边缘和执行的时间之和
                TT_edge_one = self.time_weight * edge_time_one + self.power_weight * tran_to_edge_energy_one
                TT_edge_two = self.time_weight * edge_time_two + self.power_weight * tran_to_edge_energy_two
                comm_time_to_edge_energy_one = 0
                comm_time_to_edge_energy_two = 0
                for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值 getoutput_index[m]位前驱节点的索引值
                    print(m)
                    if self.offload_two[getoutput_index[m]] == 0:  # 其前驱节点在本地执行
                        # 其前驱节点的结果从本地上传到边缘所消耗的时间
                        comm_time_to_edge_one = getoutput[getoutput_index[m]] / self.get_up_rate()  # 计算中间结果传输到边缘所需要的时间
                        comm_time_to_edge_energy_one += comm_time_to_edge_one * self.p_send

                        comm_time_to_edge_two = getoutput[
                                                    getoutput_index[m]] / self.get_up_rate_two()  # 计算中间结果传输到边缘所需要的时间
                        comm_time_to_edge_energy_two += comm_time_to_edge_two * self.p_send_two
                        comm_time_to_edge_one_temp = comm_time_to_edge_one * self.time_weight + self.power_weight * comm_time_to_edge_energy_one
                        comm_time_to_edge_two_temp = comm_time_to_edge_two * self.time_weight + self.power_weight * comm_time_to_edge_energy_two
                        if comm_time_to_edge_one_temp < comm_time_to_edge_two_temp:
                            comm_time_to_edge = comm_time_to_edge_one
                            comm_time_to_edge_flag = 1  # 表示选择的前驱任务的发射功率选的是500
                        else:
                            comm_time_to_edge = comm_time_to_edge_two
                            comm_time_to_edge_flag = 2  # 表示选择的前驱任务的发射功率选择的是1000
                        temp_commTime_LocalToEdge['LocalToEdge' + str(m)] = comm_time_to_edge
                if temp_commTime_LocalToEdge:
                    # 任务i在边缘的总时间为：任务i卸载到边缘的时间 + 任务i与其前驱节点的通信时间
                    if comm_time_to_edge_flag == 1:  # 其前驱节点的发射到边缘的功率选择的是500
                        edge_time_one_temp = edge_time_one + max(temp_commTime_LocalToEdge.values())
                        edge_time_two_temp = edge_time_two + max(temp_commTime_LocalToEdge.values())
                        TT_edge_one_temp = edge_time_one_temp * self.time_weight + (
                                tran_to_edge_energy_one + comm_time_to_edge_energy_one) * self.power_weight
                        TT_edge_two_temp = edge_time_two_temp * self.time_weight + (
                                tran_to_edge_energy_two + comm_time_to_edge_energy_one) * self.power_weight
                    else:  # 其前节点的发射到边缘的功率选择的是1000
                        edge_time_one_temp = edge_time_one + max(temp_commTime_LocalToEdge.values())
                        edge_time_two_temp = edge_time_two + max(temp_commTime_LocalToEdge.values())
                        TT_edge_one_temp = edge_time_one_temp * self.time_weight + (
                                tran_to_edge_energy_one + comm_time_to_edge_energy_two) * self.power_weight
                        TT_edge_two_temp = edge_time_two_temp * self.time_weight + (
                                tran_to_edge_energy_two + comm_time_to_edge_energy_two) * self.power_weight
                    if TT_edge_one_temp < TT_edge_two_temp:
                        power_temp = 1
                        TT_edge = TT_edge_one_temp
                    else:
                        power_temp = 2
                        TT_edge = TT_edge_two_temp
                    if comm_time_to_edge_flag == 1:
                        if power_temp == 1:
                            print("前驱节点的发射功率为500w，任务在边缘执行，发射功率为500w")
                            edge_sum_time = edge_time_one_temp
                            edge_sum_energy = tran_to_edge_energy_one + comm_time_to_edge_energy_one
                        else:
                            print("前驱节点的发射功率为500w，任务在边缘执行，发射功率为1000w")
                            edge_sum_time = edge_time_two_temp
                            edge_sum_energy = tran_to_edge_energy_two +  comm_time_to_edge_energy_one
                    else:
                        if power_temp == 1:
                            print("前驱节点的发射功率为1000w，任务在边缘执行，发射功率为500w")
                            edge_sum_time = edge_time_one_temp
                            edge_sum_energy = tran_to_edge_energy_one + comm_time_to_edge_energy_two
                        else:
                            print("前驱节点的发射功率为1000w，任务在边缘执行，发射功率为1000w")
                            edge_sum_time = edge_time_two_temp
                            edge_sum_energy = tran_to_edge_energy_two + comm_time_to_edge_energy_two
                else:
                    if TT_edge_one < TT_edge_two:
                        edge_sum_time = edge_time_one
                        edge_sum_energy = tran_to_edge_energy_one
                    else:
                        edge_sum_time = edge_time_two
                        edge_sum_energy = tran_to_edge_energy_two
                # 任务i卸载到边缘消耗本地设备的能耗为：将任务传输到边缘的传输能耗+其在本地执行的前驱节点的中间结果传输到边缘的传输能耗

                action_reward = self.time_weight * edge_sum_time + self.power_weight * edge_sum_energy
                action_reward = round(action_reward, 4)  # 将奖励值保留小数点后四位
                print("执行时间和能耗分别为")
                print(edge_sum_time)
                print(edge_sum_energy)
        # print(action_reward)
        if action_reward == self.Action_Reward[i]:
            # print(self.Action_Reward[i])
            return 1
        else:
            # print(self.Action_Reward[i])
            return 0

    # ------------------------
    # 独立惹任务的决策和根据动作生成奖励值
    # 先计算一下每个任务的在本地执行还是本地执行要好一点
    def Independent_tasks_decision(self, i):  # 这里i表示为任务编号
        local_compute_time = self.taskList[i] / self.f_local  # 计算任务本地执行的时间  所需要的cpu周期/本地设备的计算能力
        tran_to_edge_time = self.task_size[i] / self.get_up_rate()  # 将任务传输到边缘所需要的时间
        edge_execute_time = self.taskList[i] / self.f_mec  # 计算任务在边缘执行所花费的时间
        edge_sum_time = tran_to_edge_time + edge_execute_time
        if local_compute_time < edge_sum_time:
            self.offload_independence.append(0)
            local_compute_energy = local_compute_time * self.p_m  # 如果任务在本地执行的话会消耗本地设备的计算能耗
            action_reward = self.time_weight * local_compute_time + self.power_weight * local_compute_energy
            action_reward = round(action_reward, 4)  # 保留小数点后四位
            # print(action_reward)
            self.reward.append(action_reward)
            self.exe_time_Independent.append(local_compute_time)  # 记录任务在本地执行是的时间
            self.energy_Independent.append(local_compute_energy)  # 记录任务在本地执行的能耗
        else:
            self.offload_independence.append(1)
            edge_compute_energy = tran_to_edge_time * self.p_send  # 任务在边缘执行会消耗本地设备的发射能耗
            action_reward = self.time_weight * edge_sum_time + self.power_weight * edge_compute_energy
            action_reward = round(action_reward, 4)
            # print(action_reward)
            self.reward.append(action_reward)
            self.exe_time_Independent.append(edge_execute_time)  # 记录任务在边缘执行的时间
            self.energy_Independent.append(edge_compute_energy)  # 记录任务在边缘执行所消耗的能耗
        # print(local_compute_time)
        # print(edge_sum_time)

    def Independent_tasks_action_reward(self, i, a):
        if a == 0:  # 表示网络给出的决策是在本地执行
            local_execute_time = self.taskList[i] / self.f_local  # 计算任务在本地执行的时间
            local_execute_energy = local_execute_time * self.p_m  # 任务在本地执行所消耗的能耗
            r = self.time_weight * local_execute_time + self.power_weight * local_execute_energy
        else:  # 表示网络给出的任务决策实在边缘执行
            tran_to_edge_time = self.task_size[i] / self.get_up_rate()  # 计算任务上传到边缘所需要的时间
            edge_execute = self.taskList[i] / self.f_mec  # 计算任务在边缘的执行时间
            edge_sum_time = tran_to_edge_time + edge_execute  # 任务下载到边缘所消耗的总时间
            edge_execute_energy = tran_to_edge_time * self.p_send  # 任务在边缘执行，会消耗边缘设备的发射功率
            r = self.time_weight * edge_sum_time + self.power_weight * edge_execute_energy
        r = round(r, 4)
        if r == self.Action_Reward_TWO[i]:
            return 1
        else:
            return 0

    # ----------------------------
    # 通用型任务，所有任务在本地执行所消耗的能耗与时间
    def all_local_execute_time_and_energy(self):  # 所有任务在本地执行，则不需要传输时间，以及中间依赖都不需要传输
        all_local_execute_time = 0
        for i in range(len(self.taskList)):
            all_local_execute_time = all_local_execute_time + self.taskList[i] / self.f_local
        all_local_execute_energy = all_local_execute_time * self.p_m
        return all_local_execute_time, all_local_execute_energy

    #  通用型任务全部在边缘执行所消耗的时间和能耗
    def all_edge_execute_time_and_energy(self):
        total_size = 0
        for i in range(len(self.taskList)):
            total_size = total_size + self.taskList[i]
        print("total_size")
        print(total_size)
        edge_total_execute_time = total_size / self.f_mec
        total_tran_size = 0
        for j in range(len(self.task_size)):
            total_tran_size = total_tran_size + self.task_size[j]
        print("total_tran_size")
        print(total_tran_size)
        total_tran_to_edge_time = total_tran_size / self.get_up_rate()
        offload_to_edge_energy = total_tran_to_edge_time * self.p_send  # 任务全部在边缘执行会消耗本地设备的发射功耗
        offload_to_edge_time = edge_total_execute_time + total_tran_to_edge_time
        return offload_to_edge_time, offload_to_edge_energy

    # -------------------------------------------------
    # 根据卸载策略计算所有任务的执行时间
    def sumTime(self):  # i表示任务的索引号，a是所选的动作
        for i in range(len(taskList)):
            if i == 0:  # 表示是第一个任务
                if self.offload[i] == 0:  # 表示网络所选给任务做的决策是在本地执行，则下面就是计算在本地执行的reward值
                    local_execute_time = self.taskList[i] / self.f_local  # 任务在本地执行所需要的时间
                    local_compute_energy = local_execute_time * self.p_m  # 任务i在本地执行所消耗的计算能耗
                    action_reward = self.time_weight * local_execute_time + self.power_weight * local_compute_energy
                    action_reward = round(action_reward, 4)
                    print("在本地执行所消耗的时间和能量")
                    print(local_execute_time)
                    print(local_compute_energy)
                    self.sum_time.append(local_execute_time)
                    self.sum_energy.append(local_compute_energy)
                else:  # 表示第一个任务网络做的决策是卸载到边缘服务器端执行
                    tran_to_edge_time = self.task_size[i] / self.get_up_rate()  # 计算任务上传到边缘所消耗的时间
                    edge_execute_time = self.taskList[i] / self.f_mec  # 计算任务在边缘执行所消耗的时间
                    edge_offload_time = tran_to_edge_time + edge_execute_time  # 计算卸载到边缘所花费的总时间
                    tran_to_edge_energy = tran_to_edge_time * self.p_send  # 计算任务卸载到边缘所花费的能耗
                    action_reward = self.time_weight * edge_offload_time + self.power_weight * tran_to_edge_energy
                    action_reward = round(action_reward, 4)  # 奖励值保留到小数点后4位
                    print("在边缘执行所消耗的时间和能量")
                    print(edge_offload_time)
                    print(tran_to_edge_energy)
                    self.sum_time.append(edge_execute_time)

            else:  # 表示任务不是第一个任务
                getoutput = self.outputRelationship[i]  # 获取任务i依赖的前驱节点的中间数据列表
                getoutput = np.array(getoutput)  # 转换为数组
                getoutput_index = np.flatnonzero(getoutput)  # 获取非零值的索引值列表，非零值的索引也是其前驱节点的索引
                temp_commTime_EdgeToLocal = {}
                temp_commTime_LocalToEdge = {}
                if self.offload[i] == 0:  # 表示任务i在本地执行
                    local_execute_time = self.taskList[i] / self.f_local  # 任务在本地执行所需要的时间
                    local_compute_energy = local_execute_time * self.p_m  # 任务在本地执行所消耗的能耗
                    comm_Edge_To_Local_energy = 0
                    # 下面计算任务和其前驱节点在的通信时间和通信能耗，由于任务i在本地执行，只有其前驱节点在边缘执行时才会产生通信时间和通信能耗
                    for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值 getoutput_index[m]位前驱节点的索引值
                        if self.offload[getoutput_index[m]] == 1:  # 其前驱节点在边缘执行
                            # 起算其前驱节点的输出传输到本地所需要的时间
                            commTime_EdgeToLocal = getoutput[getoutput_index[m]] / self.get_down_rate()
                            # 计算任务i和其前驱节点通信所消耗的能耗
                            comm_Edge_To_Local_energy = comm_Edge_To_Local_energy + commTime_EdgeToLocal * self.p_receive
                            temp_commTime_EdgeToLocal['EdgeToLocal' + str(m)] = commTime_EdgeToLocal
                    if temp_commTime_EdgeToLocal:  # 如果任务i存在前驱节点在边缘执行
                        local_sum_time = local_execute_time + max(
                            temp_commTime_EdgeToLocal.values())  # 任务i在本地的总执行时间为：本地执行时间+中间结果的传输时间
                    else:
                        local_sum_time = local_execute_time
                    # 任务i在本地执行所消耗的能耗为：本地计算能耗 + 和前驱节点通信能耗
                    local_sum_energy = local_compute_energy + comm_Edge_To_Local_energy
                    action_reward = self.time_weight * local_sum_time + self.power_weight * local_sum_energy
                    action_reward = round(action_reward, 4)
                    print("在本地执行的时间和能耗")
                    print(local_sum_time)
                    print(local_sum_energy)
                    self.sum_time.append(local_sum_time)
                    self.sum_energy.append(local_sum_energy)
                else:  # 任务i在边缘执行
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
                            temp_commTime_LocalToEdge['LocalToEdge' + str(m)] = commTime_LocalToEdge
                    if temp_commTime_LocalToEdge:
                        # 任务i在边缘的总时间为：任务i卸载到边缘的时间 + 任务i与其前驱节点的通信时间
                        edge_sum_time = edge_offload_time + max(temp_commTime_LocalToEdge.values())
                    else:
                        edge_sum_time = edge_offload_time
                    # 任务i卸载到边缘消耗本地设备的能耗为：将任务传输到边缘的传输能耗+其在本地执行的前驱节点的中间结果传输到边缘的传输能耗
                    edge_sum_energy = tran_to_edge_energy + comm_Local_To_Edge_energy
                    action_reward = self.time_weight * edge_sum_time + self.power_weight * edge_sum_energy
                    action_reward = round(action_reward, 4)  # 将奖励值保留小数点后四位
                    print("在边缘执行的时间和能耗")
                    print(edge_sum_time)
                    print(edge_sum_energy)
                    self.sum_time.append(edge_sum_time)
                # print(action_reward)

    # 该函数计算所有任务在本地之执行所消耗的能耗
    def all_local_execute_time_and_energy_two(self, f_local):  # f_local是本地设备的cpu频率
        all_local_execute_time = 0
        energy = []
        for i in range(len(self.taskList)):
            all_local_execute_time = all_local_execute_time + self.taskList[i] / f_local
            local_energy = math.pow(10, -27) * math.pow(f_local * 1000000, 2) * self.taskList[i] * 1000000
            energy.append(local_energy)
        return energy

    # 该函数是论文的第二个点，这里将cpu的频率设置成不同的值，再次看一下卸载决策是否发生变化
    def make_decision_two_v2(self, i):
        if i == 0:  # 这里判断第一个任务是在本地执行还是在边缘执行
            local_execute_one = self.taskList[i] / self.f_local  # 本地执行时间 f_local=500MHZ
            local_compute_energy_one = math.pow(10, -27) * math.pow(self.f_local * 1000000, 2) * self.taskList[
                i] * 1000000  # 任务i在本地执行所消耗的计算能耗
            local_execute_two = self.taskList[i] / self.f_local_two
            local_compute_energy_two = math.pow(10, -27) * math.pow(self.f_local_two * 1000000, 2) * self.taskList[
                i] * 1000000
            tran_time_one = self.task_size[i] / self.get_up_rate()  # 传输时间：任务传输到边缘的时间 用的是500的发送功率
            tran_time_two = self.task_size[i] / self.get_up_rate_two()  # 传输时间：任务传输到边缘的时间 用的是500的发送功率
            edge_execute = self.taskList[i] / self.f_mec  # 任务在边缘的执行时间
            tran_to_edge_energy_one = tran_time_one * self.p_send  # 计算任务卸载到边缘所花费的能耗
            tran_to_edge_energy_two = tran_time_two * self.p_send_two  # 计算任务卸载到边缘所花费的能耗 用的是第二个等级的功率
            edge_time_one = tran_time_one + edge_execute  # 任务传输到边缘和执行的时间之和
            edge_time_two = tran_time_two + edge_execute  # 任务传输到边缘和执行的时间之和
            TT_local_one = self.time_weight * local_execute_one + self.power_weight * local_compute_energy_one
            TT_local_two = self.time_weight * local_execute_two + self.power_weight * local_compute_energy_two
            if TT_local_one < TT_local_two:
                TT_local = TT_local_one
                flag = 1  # flag是一个标记，1表示500MHZ的cpu频率下的时间和能耗加权和小，否则就是1500MHZ更好点
            else:
                TT_local = TT_local_two
                flag = 2
            TT_edge_one = self.time_weight * edge_time_one + self.power_weight * tran_to_edge_energy_one
            TT_edge_two = self.time_weight * edge_time_two + self.power_weight * tran_to_edge_energy_two
            if TT_edge_one < TT_edge_two:
                TT_edge = TT_edge_one
                edge_flag = 1
            else:
                TT_edge = TT_edge_two
                edge_flag = 2
            if TT_local < TT_edge:
                self.offload_two.append(0)  # 在本地执行
                if flag == 1:
                    print("所选取的cpu的频率是500MHZ")
                    self.exe_time.append(local_execute_one)
                    self.exe_energy.append(local_compute_energy_one)
                else:
                    print("所选取的cpu的频率是1500MHZ")
                    self.exe_time.append(local_execute_two)
                    self.exe_energy.append(local_compute_energy_two)
            else:
                self.offload_two.append(1)  # 在边缘执行
                if edge_flag == 1:
                    print("所选取的发射功率为500")
                    self.exe_time.append(TT_edge_one)
                    self.exe_energy.append(tran_to_edge_energy_one)
                else:
                    print("所选取的发射功率为1000")
                    self.exe_time.append(TT_edge_two)
                    self.exe_energy.append(tran_to_edge_energy_two)
            return [TT_local, TT_edge]
        else:
            temp_LocalTOEdge = {}
            temp_EdgeTOLocal = {}
            local_execute_one = self.taskList[i] / self.f_local  # 本地执行时间 f_local=500MHZ
            local_compute_energy_one = math.pow(10, -27) * math.pow(self.f_local * 1000000, 2) * self.taskList[
                i] * 1000000  # 任务i在本地执行所消耗的计算能耗
            local_execute_two = self.taskList[i] / self.f_local_two
            local_compute_energy_two = math.pow(10, -27) * math.pow(self.f_local_two * 1000000, 2) * self.taskList[
                i] * 1000000
            tran_time_one = self.task_size[i] / self.get_up_rate()  # 任务i上传到边缘服务器的时间
            tran_time_two = self.task_size[i] / self.get_up_rate_two()  # 任务i上传到边缘服务器的时间
            tran_to_edge_energy_one = tran_time_one * self.p_send  # 计算任务卸载到边缘所花费的能耗
            tran_to_edge_energy_two = tran_time_two * self.p_send_two  # 计算任务卸载到边缘所花费的能耗
            edge_execute = self.taskList[i] / self.f_mec  # 任务i在边缘服务器上的执行时间
            edge_time_one = tran_time_one + edge_execute  # 任务i卸载到边缘服务器的总时间=传输时间+执行时间
            edge_time_two = tran_time_two + edge_execute  # 任务i卸载到边缘服务器的总时间=传输时间+执行时间
            # self.local.append(local_execute)  # 将任务单独放在本地执行的时间放入local列表中
            # self.edge.append(tran_time)  # 将任务单独放在边缘执行的时间放入edge列表中
            getoutput = self.outputRelationship[i]  # 获取任务i依赖的前驱节点的中间数据列表
            getoutput = np.array(getoutput)  # 转换为数组
            getoutput_index = np.flatnonzero(getoutput)  # 获取非零值的索引值列表，非零值的索引也是其前驱节点的索引
            comm_time_to_edge_energy_one = 0
            comm_time_to_edge_energy_two = 0
            comm_time_to_local_energy = 0
            for m in range(len(getoutput_index)):  # 这一步就是求出其前驱节点的索引值
                if self.offload_two[getoutput_index[m]] == 0:  # 其前驱节点在本地执
                    # print(getoutput[getoutput_index[m]])
                    # print('----索引----')
                    # print(getoutput_index[m])
                    comm_time_to_edge_one = getoutput[getoutput_index[m]] / self.get_up_rate()  # 计算中间结果传输到边缘所需要的时间
                    comm_time_to_edge_energy_one += comm_time_to_edge_one * self.p_send

                    comm_time_to_edge_two = getoutput[getoutput_index[m]] / self.get_up_rate_two()  # 计算中间结果传输到边缘所需要的时间
                    comm_time_to_edge_energy_two += comm_time_to_edge_two * self.p_send_two
                    comm_time_to_edge_one_temp = comm_time_to_edge_one * self.time_weight + self.power_weight * comm_time_to_edge_energy_one
                    comm_time_to_edge_two_temp = comm_time_to_edge_two * self.time_weight + self.power_weight * comm_time_to_edge_energy_two
                    if comm_time_to_edge_one_temp < comm_time_to_edge_two_temp:
                        comm_time_to_edge = comm_time_to_edge_one
                        comm_time_to_edge_flag = 1  # 表示选择的前驱任务的发射功率选的是500
                    else:
                        comm_time_to_edge = comm_time_to_edge_two
                        comm_time_to_edge_flag = 2  # 表示选择的前驱任务的发射功率选择的是1000
                    temp_LocalTOEdge['LocalToEdge' + str(m)] = comm_time_to_edge
                else:  # 其前驱节点在边缘执行
                    # print(getoutput[getoutput_index[m]])
                    # print('---索引----')
                    # print(getoutput_index[m])
                    comm_time_to_local = getoutput[getoutput_index[m]] / self.get_down_rate()  # 中间结果传输到本地需要的时间
                    comm_time_to_local_energy += comm_time_to_local * self.p_receive  # 计算其前驱节点将中间结果传输到该节点所消耗的能耗总和
                    temp_EdgeTOLocal['EdgeTOLocal' + str(m)] = comm_time_to_local
            # 考虑任务i在本地执行，讨论其前驱节点在边缘执行还是在本地执行
            if temp_EdgeTOLocal:  # 表示i任务的前驱节点在边缘执行,任务i在本地执行
                temp_flag = 1  # 标记该任务节点是否有前驱节点，=1表示该任务有前驱节点 且该前驱节点在边缘执行
                TT_local_one_temp = local_execute_one + max(temp_EdgeTOLocal.values())  # 选取传输最大值  本地执行的时间
                TT_local_two_temp = local_execute_two + max(temp_EdgeTOLocal.values())  # 选取传输最大值
                TT_local_one = TT_local_one_temp * self.time_weight + (
                        local_compute_energy_one + comm_time_to_local_energy) * self.power_weight
                TT_local_two = TT_local_two_temp * self.time_weight + (
                        local_compute_energy_two + comm_time_to_local_energy) * self.power_weight
                if TT_local_one < TT_local_two:
                    TT_local = TT_local_one
                    flag = 1
                else:
                    TT_local = TT_local_two
                    flag = 2
            else:  # 表示任务i的前驱节点和任务i均在本地执行
                temp_flag = 2  # =2表示该任务没有前驱节点在边缘执行, 表示前驱节点在本地执行
                TT_local_one = self.time_weight * local_execute_one + self.power_weight * local_compute_energy_one
                TT_local_two = self.time_weight * local_execute_two + self.power_weight * local_compute_energy_two
                if TT_local_one < TT_local_two:
                    TT_local = TT_local_one
                    flag = 1
                else:
                    TT_local = TT_local_two
                    flag = 2
            # 考虑任务i在边缘执行，讨论其前驱节点在边缘执行还是在本地执行
            if temp_LocalTOEdge:  # 表示i任务前驱节点有在在本地执行
                if comm_time_to_edge_flag == 1:  # 其前驱节点的发射到边缘的功率选择的是500
                    edge_time_one_temp = edge_time_one + max(temp_LocalTOEdge.values())
                    edge_time_two_temp = edge_time_two + max(temp_LocalTOEdge.values())
                    TT_edge_one_temp = edge_time_one_temp * self.time_weight + (
                            tran_to_edge_energy_one + comm_time_to_edge_energy_one) * self.power_weight
                    TT_edge_two_temp = edge_time_two_temp * self.time_weight + (
                            tran_to_edge_energy_two + comm_time_to_edge_energy_one) * self.power_weight
                else:  # 其前节点的发射到边缘的功率选择的是1000
                    edge_time_one_temp = edge_time_one + max(temp_LocalTOEdge.values())
                    edge_time_two_temp = edge_time_two + max(temp_LocalTOEdge.values())
                    TT_edge_one_temp = edge_time_one_temp * self.time_weight + (
                            tran_to_edge_energy_one + comm_time_to_edge_energy_two) * self.power_weight
                    TT_edge_two_temp = edge_time_two_temp * self.time_weight + (
                            tran_to_edge_energy_two + comm_time_to_edge_energy_two) * self.power_weight
                if TT_edge_one_temp < TT_edge_two_temp:
                    power_temp = 1
                    TT_edge = TT_edge_one_temp
                else:
                    power_temp = 2
                    TT_edge = TT_edge_two_temp
            else:  # 表示任务i的前驱节点在边缘执行，则无需添加通信时间
                comm_time_to_edge_flag = 0  # 表示其前驱节点和任务i节点之间无通信时间
                TT_edge_one_temp = edge_time_one * self.time_weight + (
                    tran_to_edge_energy_one) * self.power_weight
                TT_edge_two_temp = edge_time_two * self.time_weight + (
                    tran_to_edge_energy_two) * self.power_weight
                if TT_edge_one_temp < TT_edge_two_temp:
                    power_temp = 1
                    TT_edge = TT_edge_one_temp
                else:
                    power_temp = 2
                    TT_edge = TT_edge_two_temp
            if TT_local < TT_edge:
                self.offload_two.append(0)
                if temp_flag == 1 and flag == 1:  # 表示该节点的卸载决策是在本地执行，且前驱节点有在边缘执行的
                    print("所选的cpu的频率是500MHZ，有前驱节点在边缘执行")
                    self.exe_time.append(TT_local_one)  # 记录任务该卸载策略下执行的时间
                    self.exe_energy.append(local_compute_energy_one + comm_time_to_local_energy)  # 记录任务的卸载策略下的能耗
                if temp_flag == 1 and flag == 2:  # 表示该节点的卸载决策是在本地执行，且前驱节点有在本地执行的
                    print("所选的cpu的频率是1500MHZ，有前驱节点在边缘执行")
                    self.exe_time.append(TT_local_two)  # 记录任务该卸载策略下执行的时间
                    self.exe_energy.append(local_compute_energy_two + comm_time_to_local_energy)  # 记录任务的卸载策略下的能耗
                if temp_flag == 2 and flag == 1:  # 表示该节点的卸载决策是在本地执行，且前驱节点有在本地执行的
                    print("所选的cpu的频率是500MHZ，无前驱节点在边缘执行")
                    self.exe_time.append(TT_local_one)  # 记录任务该卸载策略下执行的时间
                    self.exe_energy.append(local_compute_energy_one)  # 记录任务的卸载策
                if temp_flag == 2 and flag == 2:  # 表示该节点的卸载决策是在本地执行，且前驱节点有在本地执行的
                    print("所选的cpu的频率是1500MHZ，无前驱节点在边缘执行")
                    self.exe_time.append(TT_local_two)  # 记录任务该卸载策略下执行的时间
                    self.exe_energy.append(local_compute_energy_two)  # 记录任务的卸载策
            else:
                self.offload_two.append(1)
                if comm_time_to_edge_flag == 0 and power_temp == 1:  # 表示任务i在边缘执行，发射功率为500，前驱节点在边缘执行
                    print("表示任务在边缘执行，发射功率为500，前驱节点在边缘执行")
                    self.exe_time.append(edge_time_one)
                    self.exe_energy.append(tran_to_edge_energy_one)
                if comm_time_to_edge_flag == 0 and power_temp == 2:  # 表示任务i在边缘执行，发射功率为1000，前驱节点在边缘执行
                    print("表示任务在边缘执行，发射功率为1000，前驱节点在边缘执行")
                    self.exe_time.append(edge_time_two)
                    self.exe_energy.append(tran_to_edge_energy_two)
                if comm_time_to_edge_flag == 1 and power_temp == 1:  # 表示任务i在边缘执行，发射功率为500，前驱节点在本地执行，且发射功率为500
                    print("表示任务在边缘执行，发射功率为500，前驱节点在本地执行，且发射功率为500")
                    self.exe_time.append(edge_time_one_temp)
                    self.exe_energy.append(tran_to_edge_energy_one + comm_time_to_edge_energy_one)
                if comm_time_to_edge_flag == 1 and power_temp == 2:  # 表示任务i在边缘执行，发射功率为1000，前驱节点在边缘执行，发射功率为500
                    print("表示任务在边缘执行，发射功率为1000，前驱节点在边缘执行，发射功率为500")
                    self.exe_time.append(edge_time_two_temp)
                    self.exe_energy.append(tran_to_edge_energy_two + comm_time_to_edge_energy_one)
                if comm_time_to_edge_flag == 2 and power_temp == 1:  # 表示任务i在边缘执行，发射功率为500，前驱节点在本地执行，且发射功率为1000
                    print("表示任务在边缘执行，发射功率为500，前驱节点在本地执行，且发射功率为1000")
                    self.exe_time.append(edge_time_one_temp)
                    self.exe_energy.append(tran_to_edge_energy_one + comm_time_to_edge_energy_two)
                if comm_time_to_edge_flag == 2 and power_temp == 2:  # 表示任务i在边缘执行，发射功率为1000，前驱节点在本地执行，且发射功率为1000
                    print("表示任务在边缘执行，发射功率为1000，前驱节点在本地执行，且发射功率为1000")
                    self.exe_time.append(edge_time_two_temp)
                    self.exe_energy.append(tran_to_edge_energy_two + comm_time_to_edge_energy_two)
            return [TT_local, TT_edge]

    # 计算任务卸载或者不卸载情况下的时间和能耗



if __name__ == '__main__':
    task_size = [12, 16, 300, 21, 380, 16, 140, 20, 12, 16, 300, 21, 380, 16, 140, 20]  # 这里是输入任务的大小，单位bit，这里将输入全部去掉四个零
    taskList = [60, 150, 60, 105, 190, 80, 70, 100, 60, 150, 60, 105, 190, 80, 70, 100]  # 这里是每个任务需要的cpu的转数单位是Mcycle
    # 这里是任务依赖的前驱节点以及前驱节点输出数据大小 去掉了四个零,这里是测不同数据大小的
    task_size2 = [120, 160, 300, 210, 380, 160, 140, 200]
    taskList2 = [120, 30, 120, 200, 120, 160, 140, 200]

    task_size3 = [1200, 1600, 300, 210, 380, 1600, 1400, 200]
    taskList3 = [1200, 300, 1200, 200, 400, 1600, 140, 2000]

    task_size4 = [600, 800, 300, 1050, 380, 800, 700, 100, 600, 800, 300, 1050, 380, 800, 700, 100]
    taskList4 = [480, 120, 480, 800, 60, 640, 560, 800, 600, 800, 300, 1050, 380, 800, 700, 100]

    task_size5 = [10, 79, 253, 87, 194, 171, 110, 393, 219, 61, 33, 334, 346, 352, 371, 106]
    taskList5 = [63, 98, 141, 158, 109, 62, 192, 129, 180, 174, 188, 168, 189, 121, 169, 82]
    # 通用型任务的依赖关系图
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
                          [0, 0.96, 0, 0, 0.96, 1.12, 1.04, 0]
                          ]
    # 链型结构的依赖关系图
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
    # 树形结构的关系依赖图
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
    # 独立任务
    outputRelationship4 = [[0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0]
                           ]
    bs = BaseStation(
        f_local=500,
        f_local_two=800,
        f_mec=5000,
        p_m=0.5,
        p_o=0.01,
        p_receive=0.05,
        p_send=0.1,
        p_send_two=0.2,
        p_AP=1,
        bandwidth=2,
        time_weight=0.5,
        power_weight=0.5,
        taskList=taskList,
        outputRelationship=outputRelationship2,
        task_size=task_size
    )
    """
   
     print("***************无资源分配情况下的卸载策略********************")
    for i in range(len(taskList)):
        print(bs.make_decision_two(i))
    print(bs.offload)
    print(bs.exe_time)
    print(bs.exe_energy)
    """
    print("***************cpu频率变化情况下的卸载策略********************")
    for i in range(len(taskList)):
        print(bs.make_decision_two_v2(i))
    print(bs.offload_two)
    print(bs.exe_time)
    print(bs.exe_energy)
    print("总的时间和能耗")
    print(sum(bs.exe_time))
    print(sum(bs.exe_energy))
    print("***************无资源分配情况下的卸载策略********************")
    for i in range(len(taskList)):
        print(bs.make_decision_two(i))
    print(bs.offload)
    print("总时间和能耗")
    print(bs.exe_time_v1)
    print(bs.exe_energy_v1)
    print(sum(bs.exe_time_v1))
    print(sum(bs.exe_energy_v1))
    '''
    print("总的执行时间")
    print(sum(bs.exe_time)*0.9)
    print(sum(bs.exe_time) * 0.8)
    print(sum(bs.exe_time) * 0.7)
    print(sum(bs.exe_time) * 0.6)
    print(sum(bs.exe_time) * 0.5)
    print(sum(bs.exe_time) * 0.4)
    print(sum(bs.exe_time) * 0.3)
    print(sum(bs.exe_time) * 0.2)
    print(sum(bs.exe_time) * 0.1)

    print("总的能耗")
    print(sum(bs.exe_energy)*0.1)
    print(sum(bs.exe_energy) * 0.2)
    print(sum(bs.exe_energy) * 0.3)
    print(sum(bs.exe_energy) * 0.4)
    print(sum(bs.exe_energy) * 0.5)
    print(sum(bs.exe_energy) * 0.6)
    print(sum(bs.exe_energy) * 0.7)
    print(sum(bs.exe_time) * 0.8)
    print(sum(bs.exe_time) * 0.9)
    print(bs.sum_time)
    print(sum(bs.sum_time))
    print(sum(bs.sum_energy))
    print(bs.offload)
    exe_time = np.array(bs.exe_time)
    exe_time = np.round(exe_time, 4)  # 将执行时间保留为小数点后四位
    print(exe_time)
    energy = np.array(bs.energy)
    energy = np.round(energy, 4)  # 任务的能耗保留为小数点后四位
    print(energy)
    print(bs.action_Reward(6, 0))  # 索引为2 4 6 应该是在本地执行
    # -------DDQN计算平均时间总和、能耗总和--------#
    print("平均时间总和")
    '
    total_time = 0
    total_energy = 0
    for ii in rnge(len(exe_time)):
        total_time = total_time + exe_time[ii]
    print(total_time)
    print("消耗的能耗总和")
    for jj in range(len(energy)):
        total_energy = total_energy + energy[jj]
    print(total_energy)
    # ------------------------------------------
    '''
    # ----------所有任务在本地执行的时间和能耗总和--------------
    '''
    print("alllocal")
    print(bs.all_local_execute_time_and_energy())
    # -----------------

    # ---------所有任务在边缘执行的能耗总和
    print("alledge")
    print(bs.all_edge_execute_time_and_energy())
    # --------------------------------
    # 验证独立任务
    '''
    '''
    for i in range(len(taskList)):
        bs.Independent_tasks_decision(i)
    print(bs.offload_independence)
    print(bs.reward)
    print(bs.Independent_tasks_action_reward(4, 1))


    # ------测试独立任务的————————————
    for ii in range(len(taskList)):
        bs.Independent_tasks_decision(ii)
    total_time_Independent = 0
    total_energy_Independent = 0
    for jj in range(len(taskList)):
        total_time_Independent = total_time_Independent + bs.exe_time_Independent[jj]
        total_energy_Independent = total_energy_Independent + bs.energy_Independent[jj]
    print(bs.energy_Independent)
    print(bs.energy)
    print("DDQN独立任务的总的执行时间和总的执行能耗为：")
    print(total_time_Independent, total_energy_Independent)
    print("独立任务的决策为")
    print(bs.offload_independence)
    print("能耗和时延的加权和为")
    print(bs.reward)
    print(bs.Independent_tasks_action_reward(7, 1))  # 验证每个任务的奖励值是否正确
    '''
