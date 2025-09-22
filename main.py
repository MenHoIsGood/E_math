from seq_dataset import SeqDataset_NFX, create_seq_dataloader
from torch import nn, optim
import torch
import pandas as pd
from model.tsmixerx_embedding import TSMixerXEmbedding


def train_epoch(model, dataloader, max_epochs):
    """
    训练模型
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 判断是否有可用的GPU
    if torch.cuda.is_available():
        # 创建一个设备对象，使用默认的GPU设备
        device = torch.device("cuda")
    else:
        # 如果没有可用的GPU，则使用CPU设备
        device = torch.device("cpu")

    model = model.to(device)

    model.train()

    for t in range(max_epochs):
        correct = 0
        avg_loss = 0
        total_samples = 0
        # 训练模型
        for data in dataloader:
            optimizer.zero_grad()
            y = data["label"]
            yp = model(data)
            # 分别对预测结果和真实值进行逐个指标的损失计算和反向传播
            pred = yp.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            loss = loss_fn(yp, y)
            avg_loss += loss.item()
            total_samples += data["main"].size(0)
            loss.backward()
            # 更新参数要写在外面，等累积完梯度后再更新
            optimizer.step()

        num_batches = len(dataloader)
        avg_loss /= num_batches
        info = ' - '.join([
            f'[Epoch {t + 1}/{max_epochs}]',
            f'loss: {avg_loss:.4f}',
            f'acc: {100 * correct / total_samples:.2f}%'
        ])
        print(info)

    print("训练完成!")

    return model

def predict(model, dataloader, group, labels, device):

    label_num = len(labels)
    true_label_index = -1
    print(f"开始对 {group} 的数据进行故障分析")
    model.eval()
    correct = 0
    total_samples = 0
    counts = torch.zeros(label_num, dtype=torch.int64)
    for data in dataloader:
        y = data["label"]
        yp = model(data)
        temp_score = yp.sum(dim=0)
        # 分别对预测结果和真实值进行逐个指标的损失计算和反向传播
        pred = yp.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total_samples += data["main"].size(0)
        true_label_index = y[0].item()
        # 遍历 pred，把对应索引的元素加1
        for val in pred.view(-1):  # flatten成一维
            counts[val] += 1

    # 找到最大值索引（如果有多个最大值，返回第一个）
    pred_idx = torch.argmax(counts)
    print(counts)
    print(f"{group} 的预测结果是 {labels[pred_idx]},真实结果是 {labels[true_label_index]}")
    print(f"{group} 的切片准确率是 {100 * correct / total_samples:.2f}%")

if __name__ == '__main__':
    labels = ['B', 'IR', 'OR', 'N']


    test_df = pd.read_csv('源域数据集重采样/test_df.csv')
    # train_df = pd.read_csv('源域数据集重采样/train_df.csv')
    # print("加载df完成！")
    #
    # train_dataset = SeqDataset_NFX(processed_data=train_df,
    #                                labels=['B','IR','OR', 'N'],
    #                                seq_length=4000,
    # )
    # print("train_dataset制作完成！")
    # train_dataloader = create_seq_dataloader(dataset=train_dataset,
    #                                          batch_size=16)
    #
    # model = TSMixerXEmbedding(
    #     h=4000,  # 这个是用不上的
    #     input_size=4000,
    #     n_series=4,
    # )
    #
    # print("开始训练")
    # train_model = train_epoch(model, train_dataloader, max_epochs=10)
    # torch.save(train_model.state_dict(), 'model.pth')
    # print(f"模型保存成功，保存在 model.pth 中")

    # 测试部分
    print("开始测试")
    test_model = TSMixerXEmbedding(
        h=4000,  # 这个是用不上的
        input_size=4000,
        n_series=4,
    )
    # 判断是否有可用的GPU
    if torch.cuda.is_available():
        # 创建一个设备对象，使用默认的GPU设备
        device = torch.device("cuda")
    else:
        # 如果没有可用的GPU，则使用CPU设备
        device = torch.device("cpu")
    test_model = test_model.to(device)
    test_model.load_state_dict(torch.load("model.pth"))


    test_dfs = []
    groups = ['FE_OR021@3_2', 'FE_B007_3', 'DE_OR007@3_1', 'DE_OR007@12_0', 'FE_IR014_1',
              'DE_IR007_1', 'FE_B014_3', 'FE_IR014_0', 'DE_B028_1', 'DE_OR021@12_0',
              'DE_IR021_0', 'DE_B028_0', 'DE_OR021@12_2', 'FE_OR014@6_0', 'DE_OR007@6_2',
              'DE_OR021@6_2', 'FE_OR021@6_0', 'FE_IR007_2', 'DE_B021_2', 'FE_OR007@3_2',
              'FE_OR021@3_1', 'N_1']
    # 按照指定的 groups 将 test_df 分割成多个 DataFrame
    for group in groups:
        # 筛选出 group 列等于指定 group 的行
        group_df = test_df[test_df['group'] == group].copy()

        # 如果存在该 group 的数据，则添加到 test_dfs 中
        if not group_df.empty:
            test_dfs.append(group_df)
        else:
            print(f"警告: group '{group}' 在 test_df 中未找到数据")

    print(f"成功分割出 {len(test_dfs)} 个测试数据集")

    for df in test_dfs:
        group = df['group'].unique()[0]
        test_dataset = SeqDataset_NFX(processed_data=df,
                                      labels=labels,
                                      seq_length=4000,
                                      )
        test_dataloader = create_seq_dataloader(dataset=test_dataset,
                                                batch_size=16)
        predict(test_model, test_dataloader, group, labels, device)
