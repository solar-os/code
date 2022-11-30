import numpy as np
import math



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
        self.offload = []
        self.exe_time = []
    # 用户上传数据到基站的数据传输速率
    def get_up_rate(self):
        #  print(2000000*math.log2(1+math.pow(10, 9)/math.pow(20,4)))
        return 8000000  #  b/s  这里数据的大小用kb单位

    # 用户从基站下载数据的传输速率
    def get_down_rate(self):
        return 1150000  # 用户下载时的传输速率为 b/s

    # 判断任务是本地执行还是边缘执行,卸载决策存储在offload列表中，执行时间存储在exe_time列表中
    def make_decision(self, i):

        if i == 0:
            local_execute = self.taskList[i] / self.f_local  # 本地执行时间
            tran_time = self.task_size[i] / self.get_up_rate()  # 传输时间：任务传输到边缘的时间
            edge_execute = self.taskList[i] / self.f_mec  # 任务在边缘的执行时间
            edge_time = tran_time + edge_execute   # 任务传输到边缘和执行的时间之和
            if local_execute < edge_time:
                self.offload.append(0)
                self.exe_time.append(local_execute)
                return [local_execute, edge_time]
            else:
                self.offload.append(1)
                self.exe_time.append(edge_time)
                return [local_execute, edge_time]
        else:
            local_execute = self.taskList[i] / self.f_local  # 任务i在本地执行所需要的时间
            tran_time = self.task_size[i] / self.get_up_rate()
            edge_execute = self.taskList[i] / self.f_mec
            edge_time = tran_time + edge_execute
            if self.offload[i-1] == 0:
                getoutput = self.outputRelationship[i]  # 获取依赖的前驱节点的数据
                getoutput = np.array(getoutput)
                getoutput = getoutput[np.nonzero(getoutput)]
                comm_time_to_edge = max(getoutput) / self.get_up_rate()  # 中间结果传输到边缘所需要的时间
                edge_time = edge_time + comm_time_to_edge  # 任务i在边缘执行需要的时间
                if local_execute < edge_time:
                    self.offload.append(0)
                    self.exe_time.append(local_execute)
                    return [local_execute, edge_time]
                else:
                    self.offload.append(1)
                    self.exe_time.append(edge_time)
                    return [local_execute, edge_time]
            elif self.offload[i-1] == 1:
                getoutput = self.outputRelationship[i]
                getoutput = np.array(getoutput)
                getoutput = getoutput[np.nonzero(getoutput)]
                comm_time_to_local = max(getoutput) / self.get_down_rate()  # 中间结果传输到本地需要的时间
                local_execute = local_execute + comm_time_to_local
                if local_execute < edge_time:
                    self.offload.append(0)
                    self.exe_time.append(local_execute)
                    return [local_execute, edge_time]
                else:
                    self.offload.append(1)
                    self.exe_time.append(edge_execute)
                    return [local_execute, edge_time]

    # 通过读取卸载决策offload列表和执行时间列表exe_time，计算每个任务节点的能量消耗，这里的i为任务的索引号
    def energy_consumption(self, i):
        pass
    def local_power(self, task):
        return self.local_time(task) * self.pm

    def edge_time(self, task):
        return task[0] * self.C / self.Scpu_speed + task[0] / self.TXup + \
               task[1] / self.TXdown  # 计算卸载时间= 传输时间+执行时间+等待时间

    def edge_power(self, task):
        return task[0] / self.TXup * self.pup + task[1] / self.TXdown * self.pdown

    def get_myuser_profit(self, i):
        local_time = self.local_time(i)
        local_power = self.local_power(i)
        edge_time = self.edge_time(i)
        edge_power = self.edge_power(i)
        user_profit = (self.time_weight * local_time + self.power_weight * local_power) - (
                self.time_weight * edge_time + self.power_weight * edge_power)
        return user_profit

    def make_mydecision(self, i):
        if self.get_myuser_profit(i) > 0:
            return 'edge'
        else:
            return 'local'

    def cost(self, i):
        if self.make_mydecision(i) == 'edge':
            return -1 * (self.time_weight * self.edge_time(i) + self.power_weight * self.edge_power(i))
        else:
            return -1 * (self.time_weight * self.local_time(i) + self.power_weight * self.local_power(i))

    def local_cost(self, task):
        if task[2] > self.cpu_speed:  # 如果选择本地执行时，如果移动设备的cpu小于我们的任务需要的CPU 则设置很小的奖励作为惩罚
            return -1000
        else:
            return -1 * (self.time_weight * self.local_time(task) + self.power_weight * self.local_power(task))

    def edge_cost(self, task):
        return -1 * (self.time_weight * self.edge_time(task) + self.power_weight * self.edge_power(task))

    def local_costT(self, task):
        # if task[2] > self.cpu_speed:  # 如果选择本地执行时，如果移动设备的cpu小于我们的任务需要的CPU 则设置很小的奖励作为惩罚
        # return -1000
        # else:
        return self.time_weight * self.local_time(task) + self.power_weight * self.local_power(task)

    def edge_costT(self, task):
        return self.time_weight * self.edge_time(task) + self.power_weight * self.edge_power(task)




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
        print(bs.make_decision(i))
    print(bs.offload)
    print(bs.exe_time)