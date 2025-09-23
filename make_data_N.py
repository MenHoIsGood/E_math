import scipy.io as sio
import numpy as np
import pandas as pd

dfs = []



for i in range(4):

    raw = sio.loadmat(f'源域数据集重采样/48kHz_Normal_data-_32kHz_Normal_data/N_{i}.mat')
    keys = raw.keys()
    # 找到以 '_DE_time' 结尾的键名
    de_time_keys = [key for key in keys if key.endswith('_DE_time')]
    de_time_key = de_time_keys[0] if de_time_keys else None
    # 找到以 '_FE_time' 结尾的键名
    fe_time_keys = [key for key in keys if key.endswith('_FE_time')]
    fe_time_key = fe_time_keys[0] if fe_time_keys else None
    # dict_keys(['__header__', '__version__', '__globals__', 'X118_DE_time', 'X118_FE_time', 'X118_BA_time', 'X118RPM'])
    temp_df = pd.DataFrame({
        'main': 0 if fe_time_key == None else raw[fe_time_key].flatten(),
        'sup': 0 if de_time_key == None else raw[de_time_key].flatten(),
        'error_type': None,
        'load_force': f'{i}',
        'error_size': 0,
        'group': f'N_{i}',
        'label': 'N'
    })
    temp_df['time'] = range(len(temp_df))
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)

df.to_csv('源域数据集重采样/N.csv', index=False)