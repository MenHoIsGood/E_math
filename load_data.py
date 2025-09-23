import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 读取 .mat 文件
data = sio.loadmat('数据集/目标域数据集/A.mat')

# 查看文件中有哪些变量
# 目标域数据集A
# print(data.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'A'])
# # 假设你需要的变量叫 'A'
# X = data['A']
#
# # 转换成 PyTorch Tensor
# X_tensor = torch.from_numpy(X).float()
# print(X_tensor.shape)
# torch.Size([256000, 1])

# 查看文件中有哪些变量
# 源域'数据集/源域数据集/48kHz_Normal_data/N_0.mat'
print(data.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'X097_DE_time', 'X097_FE_time', 'X097RPM'])

X1 = data['A']
RPM = 1800
fr = RPM/60

DE_n = 9
DE_d = 0.3126
DE_D = 1.537

FE_n = 9
FE_d = 0.2656
FE_D = 1.122

BPFO = np.ceil(fr*DE_n/2*(1-DE_d/DE_D))


BSF = np.ceil(fr*DE_D/DE_d*(1-(DE_d/DE_D)*(DE_d/DE_D)))
T = 4000

# 从 X1 中取出前 3T 行数据
X1_first_T_rows = X1[:T, :]  # 取前 3T 行的所有列数据

# 方法1：使用 matplotlib 绘制折线图
plt.figure(figsize=(100, 6))
plt.plot(X1_first_T_rows[:, 0])  # 取第一列数据进行绘制
plt.xlabel('Index (N axis)')
plt.ylabel('Value (1 axis)')
plt.title('Line Plot of (N, 1) shaped ndarray')
plt.grid(True)
plt.show()
