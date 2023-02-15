import csv
import random
import data_poison_attack_Normal
import numpy as np
import pandas as pd

from data_poison_attack_Normal.attack import attack
from data_poison_attack_model.attack_recommend import attack_recommend
from recommend_system.recommend_system import recommend

if __name__ == '__main__':
    # 读取数据集
    # 数据预处理
    ratings = pd.read_csv('./dataset/ratings.csv', parse_dates=['timestamp'])
    # 取十分之三的数据
    rand_userIds = np.random.choice(ratings['userId'].unique(),
                                    size=int(len(ratings['userId'].unique())),
                                    replace=False)
    ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]
    ratings.sample(5)
    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

    # 随机选取正常用户用户
    # userIds = [random.randint(0, 5000) for i in range(5)]
    userIds = [1, 406, 157, 5773]

    # 划分为训练集和测试集 取最新的数据作为训练集
    train_ratings = ratings[ratings['rank_latest'] != 1]
    # 删除我们不再需要的列
    train_ratings = train_ratings[['userId', 'movieId', 'rating']]
    # 将显式反馈变为隐式反馈
    train_ratings.loc[:, 'rating'] = 1
    train_ratings.sample(5)
    # 生成推荐系统并为其推荐top10
    recommend(userIds, ratings, train_ratings)

    # 攻击方式1：随机插入相关数据
    attack(ratings)

    # 重新创建推荐系统并重新生成top10与之间对应
    # 划分为训练集和测试集 取最新的数据作为训练集
    train_ratings = ratings[ratings['rank_latest'] != 1]
    # 删除我们不再需要的列
    train_ratings = train_ratings[['userId', 'movieId', 'rating']]
    # 将显式反馈变为隐式反馈
    train_ratings.loc[:, 'rating'] = 1
    train_ratings.sample(5)
    recommend(userIds, ratings, train_ratings)

    # 攻击方式2：使用代理模型生成配置文件
    train_ratings = ratings[ratings['rank_latest'] != 1]
    train_ratings = train_ratings[['userId', 'movieId', 'rating']]
    train_ratings.loc[:, 'rating'] = 1
    train_ratings.sample(5)
    ratings = attack_recommend(ratings, train_ratings)
    # 重新创建推荐系统并重新生成top10与之间对应
    train_ratings = ratings[ratings['rank_latest'] != 1]
    train_ratings = train_ratings[['userId', 'movieId', 'rating']]
    train_ratings.loc[:, 'rating'] = 1
    train_ratings.sample(5)
    recommend(userIds, ratings, train_ratings)
