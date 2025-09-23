import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple


class SeqDataset_NFX(Dataset):
    """
        Dataset
    """

    def __init__(self, processed_data: pd.DataFrame,
                 main_col: str = 'main',
                 sup_col: str = 'sup',
                 label_col: str = 'label',
                 labels=None,
                 seq_length: int = 4000,
                 group_column: str = 'group',
                 date_column: str = 'time'
                 ):

        if labels is None:
            labels = ['B', 'IR', 'OR', 'N']
        self.seq_length = seq_length
        self.date_column = date_column
        self.group_column = group_column
        self.label_col = label_col
        self.main_col = main_col
        self.sup_col = sup_col

        self.data = processed_data
        # 创建标签到索引的映射字典
        label_to_index = {label: idx for idx, label in enumerate(labels)}

        # 将 label_col 列的数据映射成 0 到 len(labels)-1 的值
        self.data[label_col] = self.data[label_col].map(label_to_index)


        # 判断是否有可用的GPU
        if torch.cuda.is_available():
            # 创建一个设备对象，使用默认的GPU设备
            self.device = torch.device("cuda")
            print(f"使用GPU设备: {self.device}")
        else:
            # 如果没有可用的GPU，则使用CPU设备
            self.device = torch.device("cpu")
            print(f"使用CPU设备: {self.device}")

        # 构建滑动窗口数据
        self.sequences = self._create_sequences()
        print(f"构建了 {len(self.sequences)} 个有效序列")


    def _create_sequences(self):
        """创建所有有效的序列"""
        sequences = []

        df = self.data.copy()

        unique_groups = df[self.group_column].unique().tolist()

        for group in unique_groups:  # 按组织分序列
            # 从 df 中筛选出 group_column 列等于 group 的行
            group_df = df[df[self.group_column] == group].copy()
            group_df = group_df.reset_index(drop=True)

            # 创建滑动窗口
            seq_length = self.seq_length
            padding = seq_length / 2
            for start in range(0, len(group_df), int(padding)):  # 滑动窗口步长为 padding
                # 筛选出 time >= start 且 time < start + padding 的数据
                if start + seq_length > len(group_df):
                    continue
                mask = (group_df[self.date_column] >= start) & (group_df[self.date_column] < start + seq_length)
                filtered_data = group_df[mask].copy()

                # 按 time 列从小到大排序
                filtered_data = filtered_data.sort_values(by=self.date_column).reset_index(drop=True)

                # 如果有数据则处理
                if len(filtered_data) > 0:
                    # 提取筛选和排序后的数据
                    seq_main = filtered_data[self.main_col].values.astype(np.float32)
                    seq_sup = filtered_data[self.sup_col].values.astype(np.float32)
                    seq_label = int(filtered_data[self.label_col].iloc[0])  # 取第一个元素并转成 int

                    # 创建张量,放到显存中
                    seq_main_tensor = (torch.FloatTensor(seq_main)).to(self.device)
                    seq_sup_tensor = (torch.FloatTensor(seq_sup)).to(self.device)
                    seq_label_tensor = torch.tensor(seq_label, dtype=torch.int64, device=self.device)


                    sequences.append({
                        'main': seq_main_tensor,
                        'sup': seq_sup_tensor,
                        'label': seq_label_tensor,
                    })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        获取一个序列样本
        """
        seq_data = self.sequences[idx]

        return {
            'main': seq_data['main'],
            'sup': seq_data['sup'],
            'label': seq_data['label'],
        }


def create_seq_dataloader(dataset: SeqDataset_NFX, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    创建序列数据加载器
    """

    def collate_fn(batch):
        main = torch.stack([item['main'] for item in batch])
        sup = torch.stack([item['sup']for item in batch])
        label = torch.stack([item['label'] for item in batch])

        return {
            'main':main,
            'sup':sup,
            'label':label,
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
