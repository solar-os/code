"""
随机生成多个用户，每个用户着需要执行的依赖任务，依赖任务有三个属性:
task_size表示每个子任务的数据大小
taskList表示每个任务所需要的CPU的周期
outputRelationship表示子任务的前后依赖关系，这是一个列表，具体的数值表示前一个任务向后一个任务传递的数据大小
"""
import random
import math
f = open("userTaskSize.txt", "w")
print((2000000*math.log2(1+math.pow(10, 9)/math.pow(20, 4))))
print((2000000*math.log2(1+2*math.pow(10, 9)/math.pow(20, 4)))/(2000000*math.log2(1+math.pow(10, 9)/math.pow(20, 4))))
task_size_temp = []
task_size = []
task_list_temp = []
tasK_list = []
for i in range(10):  # 这里表示10个用户
    for j in range(16):  # 这里表示一个任务由16个子任务组成
        task_size_temp.append(random.randint(10, 400))  # 这里临时存储每个用户的任务数据大小
    task_size_temp = []
    task_size.append(task_size_temp)
    f.write()
print(task_size)
for m in range(10):  # 这里表示10个用户
    for n in range(16):  # 这里表示一个任务由16个子任务组成
        task_list_temp.append(random.randint(60, 200))  # 这里临时存储每个用户所需要的cpu的转数，单位为Mcycle
    task_list_temp = []
    tasK_list.append(task_list_temp)
print(tasK_list)
