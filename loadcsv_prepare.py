import numpy as np
import pandas as pd

OR_list = ['Centered','Opposite','Orthogonal']

error_list = ['B','IR','OR']
type_list = ['DE','FE']
load_list = ['0', '1', '2', '3']
labels = ['B','IR','OR', 'N']
size_list = [0, 7, 14, 21, 28]

dfs = []
for error in error_list:
    for type in type_list:
        if error == 'OR':
            for OR in OR_list:
                df = pd.read_csv('源域数据集重采样/%s_%s_%s.csv' % (error, OR, type))
                dfs.append(df)
        else:
            df = pd.read_csv('源域数据集重采样/%s_%s.csv' % (error, type))
            dfs.append(df)

df = pd.read_csv('源域数据集重采样/N.csv')
dfs.append(df)

raw_df = pd.concat(dfs, ignore_index=True)

unique_groups = raw_df['group'].unique().tolist()

# 选择 20% 的 group（四舍五入）
selected_groups = ['FE_OR021@3_2','FE_B007_3','DE_OR007@3_1','DE_OR007@12_0','FE_IR014_1',
 'DE_IR007_1','FE_B014_3','FE_IR014_0', 'DE_B028_1', 'DE_OR021@12_0',
 'DE_IR021_0', 'DE_B028_0', 'DE_OR021@12_2', 'FE_OR014@6_0', 'DE_OR007@6_2',
 'DE_OR021@6_2', 'FE_OR021@6_0', 'FE_IR007_2', 'DE_B021_2', 'FE_OR007@3_2',
 'FE_OR021@3_1','N_1']


# 划分数据
mask = raw_df['group'].isin(selected_groups)
df_selected = raw_df[mask].copy()      # 选中的 20% group 数据
df_remaining = raw_df[~mask].copy()    # 剩余的 80% group 数据

df_selected.to_csv('源域数据集重采样/test_df.csv', index=False)
df_remaining.to_csv('源域数据集重采样/train_df.csv', index=False)


# 显示划分结果
print(f"总 group 数: {len(unique_groups)}")
print(f"选中 group 数: {len(selected_groups)} ({len(selected_groups)/len(unique_groups)*100:.1f}%)")
print(f"选中数据行数: {len(df_selected)}")
print(f"剩余数据行数: {len(df_remaining)}")

# 定义
RPM = 1800 # 固定为1800转速
fr = RPM/60

DE_n = 9
DE_d = 0.3126
DE_D = 1.537

FE_n = 9
FE_d = 0.2656
FE_D = 1.122

# 外圈故障频率（向上取整）
BPFO_DE = 108
BPFO_FE = 104

# 内圈故障(就是fr)
fr = 30

# 滚动体公转频率(向上取整)
FTF_DE  = 12
FTF_FE  = 12

# 目标域
FTF = 4

# 32k除以8是4000，为了方便计算，选择4000个采样点为一个采样样本范围，滑动窗口每次滑动2000个采样点，
T=8000