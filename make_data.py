import scipy.io as sio
import numpy as np
import pandas as pd


dfs = []
sizes = [['0007','B007'], ['0014', 'B014'], ['0021', 'B021'], ['0028', 'B028']]

for size, version in sizes:
    for i in range(4):

        raw = sio.loadmat(f'源域数据集重采样/12kHz_DE_data-_32kHz_DE_data/B/{size}/{version}_{i}.mat')
        keys = raw.keys()
        # 找到以 '_DE_time' 结尾的键名
        de_time_keys = [key for key in keys if key.endswith('_DE_time')]
        de_time_key = de_time_keys[0] if de_time_keys else None
        # 找到以 '_FE_time' 结尾的键名
        fe_time_keys = [key for key in keys if key.endswith('_FE_time')]
        fe_time_key = fe_time_keys[0] if fe_time_keys else None
        # dict_keys(['__header__', '__version__', '__globals__', 'X118_DE_time', 'X118_FE_time', 'X118_BA_time', 'X118RPM'])
        temp_df = pd.DataFrame({
            'main': 0 if de_time_key == None else raw[de_time_key].flatten(),
            'sup': 0 if fe_time_key == None else raw[fe_time_key].flatten(),
            'error_type': 'DE',
            'load_force': f'{i}',
            'error_size': size,
            'group': f'DE_{version}_{i}',
            'label': 'B'
        })
        temp_df['time'] = range(len(temp_df))
        dfs.append(temp_df)

B_DE_df = pd.concat(dfs, ignore_index=True)

B_DE_df.to_csv('源域数据集重采样/B_DE.csv', index=False)


dfs = []
sizes = [['0007','IR007'], ['0014', 'IR014'], ['0021', 'IR021'], ['0028', 'IR028']]

for size, version in sizes:
    for i in range(4):

        raw = sio.loadmat(f'源域数据集重采样/12kHz_DE_data-_32kHz_DE_data/IR/{size}/{version}_{i}.mat')
        keys = raw.keys()
        # 找到以 '_DE_time' 结尾的键名
        de_time_keys = [key for key in keys if key.endswith('_DE_time')]
        de_time_key = de_time_keys[0] if de_time_keys else None
        # 找到以 '_FE_time' 结尾的键名
        fe_time_keys = [key for key in keys if key.endswith('_FE_time')]
        fe_time_key = fe_time_keys[0] if fe_time_keys else None
        # dict_keys(['__header__', '__version__', '__globals__', 'X118_DE_time', 'X118_FE_time', 'X118_BA_time', 'X118RPM'])
        temp_df = pd.DataFrame({
            'main': 0 if de_time_key == None else raw[de_time_key].flatten(),
            'sup': 0 if fe_time_key == None else raw[fe_time_key].flatten(),
            'error_type': 'DE',
            'load_force': f'{i}',
            'error_size': size,
            'group': f'DE_{version}_{i}',
            'label': 'IR'
        })
        temp_df['time'] = range(len(temp_df))
        dfs.append(temp_df)

IR_DE_df = pd.concat(dfs, ignore_index=True)
IR_DE_df.to_csv('源域数据集重采样/IR_DE.csv', index=False)

dfs = []
sizes = [['0007','OR007@6'], ['0014', 'OR014@6'], ['0021', 'OR021@6']]

for size, version in sizes:
    for i in range(4):

        raw = sio.loadmat(f'源域数据集重采样/12kHz_DE_data-_32kHz_DE_data/OR/Centered/{size}/{version}_{i}.mat')
        keys = raw.keys()
        # 找到以 '_DE_time' 结尾的键名
        de_time_keys = [key for key in keys if key.endswith('_DE_time')]
        de_time_key = de_time_keys[0] if de_time_keys else None
        # 找到以 '_FE_time' 结尾的键名
        fe_time_keys = [key for key in keys if key.endswith('_FE_time')]
        fe_time_key = fe_time_keys[0] if fe_time_keys else None
        # dict_keys(['__header__', '__version__', '__globals__', 'X118_DE_time', 'X118_FE_time', 'X118_BA_time', 'X118RPM'])
        temp_df = pd.DataFrame({
            'main': 0 if de_time_key == None else raw[de_time_key].flatten(),
            'sup': 0 if fe_time_key == None else raw[fe_time_key].flatten(),
            'error_type': 'DE',
            'load_force': f'{i}',
            'error_size': size,
            'group': f'DE_{version}_{i}',
            'label': 'OR'
        })
        temp_df['time'] = range(len(temp_df))
        dfs.append(temp_df)

IR_DE_df = pd.concat(dfs, ignore_index=True)
IR_DE_df.to_csv('源域数据集重采样/OR_Centered_DE.csv', index=False)

dfs = []
sizes = [['0007','OR007@12'], ['0021', 'OR021@12']]

for size, version in sizes:
    for i in range(4):

        raw = sio.loadmat(f'源域数据集重采样/12kHz_DE_data-_32kHz_DE_data/OR/Opposite/{size}/{version}_{i}.mat')
        keys = raw.keys()
        # 找到以 '_DE_time' 结尾的键名
        de_time_keys = [key for key in keys if key.endswith('_DE_time')]
        de_time_key = de_time_keys[0] if de_time_keys else None
        # 找到以 '_FE_time' 结尾的键名
        fe_time_keys = [key for key in keys if key.endswith('_FE_time')]
        fe_time_key = fe_time_keys[0] if fe_time_keys else None
        # dict_keys(['__header__', '__version__', '__globals__', 'X118_DE_time', 'X118_FE_time', 'X118_BA_time', 'X118RPM'])
        temp_df = pd.DataFrame({
            'main': 0 if de_time_key == None else raw[de_time_key].flatten(),
            'sup': 0 if fe_time_key == None else raw[fe_time_key].flatten(),
            'error_type': 'DE',
            'load_force': f'{i}',
            'error_size': size,
            'group': f'DE_{version}_{i}',
            'label': 'OR'
        })
        temp_df['time'] = range(len(temp_df))
        dfs.append(temp_df)

IR_DE_df = pd.concat(dfs, ignore_index=True)
IR_DE_df.to_csv('源域数据集重采样/OR_Opposite_DE.csv', index=False)



dfs = []
sizes = [['0007','OR007@3'], ['0021', 'OR021@3']]

for size, version in sizes:
    for i in range(4):

        raw = sio.loadmat(f'源域数据集重采样/12kHz_DE_data-_32kHz_DE_data/OR/Orthogonal/{size}/{version}_{i}.mat')
        keys = raw.keys()
        # 找到以 '_DE_time' 结尾的键名
        de_time_keys = [key for key in keys if key.endswith('_DE_time')]
        de_time_key = de_time_keys[0] if de_time_keys else None
        # 找到以 '_FE_time' 结尾的键名
        fe_time_keys = [key for key in keys if key.endswith('_FE_time')]
        fe_time_key = fe_time_keys[0] if fe_time_keys else None
        # dict_keys(['__header__', '__version__', '__globals__', 'X118_DE_time', 'X118_FE_time', 'X118_BA_time', 'X118RPM'])
        temp_df = pd.DataFrame({
            'main': 0 if de_time_key == None else raw[de_time_key].flatten(),
            'sup': 0 if fe_time_key == None else raw[fe_time_key].flatten(),
            'error_type': 'DE',
            'load_force': f'{i}',
            'error_size': size,
            'group': f'DE_{version}_{i}',
            'label': 'OR'
        })
        temp_df['time'] = range(len(temp_df))
        dfs.append(temp_df)

IR_DE_df = pd.concat(dfs, ignore_index=True)
IR_DE_df.to_csv('源域数据集重采样/OR_Orthogonal_DE.csv', index=False)