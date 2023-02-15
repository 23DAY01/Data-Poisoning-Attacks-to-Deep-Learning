import warnings
import random

import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch数据集用于训练

    Args:
        ratings (pd.DataFrame): 包含电影评级的DataFrame
        all_movieIds (list): 包含所有电影id的列表

    """

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    # 获取数据集
    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['userId'], ratings['movieId']))

        # 按照4:1进行负采样
        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            # 负采样
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class NCF(pl.LightningModule):
    """ 神经协同过滤(NCF)

        Args:
            num_users (int): 唯一用户的数量
            num_items (int): 唯一项的数量
            ratings (pd.DataFrame): 包含用于训练的电影评级
            all_movieIds (list): 包含所有movieIds的列表(训练+测试)
    """

    def __init__(self, num_users, num_items, ratings, all_movieIds):
        super().__init__()
        # 用八个维度嵌入
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds

    # 预测 通过用户输入与项目输入输出预测值
    def forward(self, user_input, item_input):
        # 通过嵌入层
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        # Concat两个嵌入层
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        # 通过全连接层
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        # 输出层
        pred = nn.Sigmoid()(self.output(vector))
        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                          batch_size=512, num_workers=0)


def attack_recommend(ratings, train_ratings):
    warnings.filterwarnings('ignore')
    # 设置关键参数 目标项目id 攻击配置文件数量 目标评分
    target_movie_id = 99
    target_rate = 5
    last_user_id = ratings.loc[ratings['userId'].idxmax()].userId

    num_users = ratings['userId'].max() + 1
    num_items = ratings['movieId'].max() + 1
    all_movieIds = ratings['movieId'].unique()
    random_userId = [random.randint(0, 100) for i in range(100)]
    attack_userId = []
    for ran in random_userId:
        attack_userId.append(ran + 2000)

    # 建立ncf模型
    model = NCF(num_users, num_items, train_ratings, all_movieIds)

    # 开始训练
    trainer = pl.Trainer(max_epochs=1, gpus=1, reload_dataloaders_every_n_epochs=1, logger=False,
                         enable_checkpointing=False, accelerator='cpu')
    trainer.fit(model)

    # 每个用户与之交互的所有条目
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()
    # print(user_interacted_items)

    # 通过计算给定用户的top10从而选择他们的配置文件
    for u in attack_userId:
        # interacted_items = user_interacted_items[406]
        # not_interacted_items = set(all_movieIds) - set(interacted_items)
        # # 选取没有交互过的100个项目再加目标项
        # selected_not_interacted = list(np.random.choice(list(not_interacted_items), 100))
        # test_items = selected_not_interacted
        items = [random.randint(0, 1000) for i in range(100)]
        # 预测标签
        predicted_labels = np.squeeze(model(torch.tensor([u] * 100),
                                            torch.tensor(items)).detach().numpy())

        top10_items = [items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
        for top10 in top10_items:
            new_row = {'userId': u,
                       'movieId': top10,
                       'rating': target_rate,
                       'timestamp': 978301968}
            ratings.append(new_row, ignore_index=True)
        new_row = {'userId': u,
                   'movieId': target_movie_id,
                   'rating': target_rate,
                   'timestamp': 978301968}
        ratings.append(new_row, ignore_index=True)
    return ratings
