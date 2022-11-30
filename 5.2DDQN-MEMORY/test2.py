import numpy as np
import math
a = [2, 0, 3]
a = np.array(a)
#print(np.flatnonzero(a))  # 获取非零值的索引值
a = a[np.nonzero(a)]  # 获取非零值
temp ={}
temp['localtoedge'] = 12
temp['edgetolocal'] = 15
print(max(temp, key=temp.get)[:-1])  # 获取字典中最大值的键

print(max(temp.values()))       # 获取字典中的最大值
for i in range(8):
    print(i)