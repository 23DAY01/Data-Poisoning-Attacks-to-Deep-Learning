import warnings

import pandas as pd


def attack(df_ratings):
    warnings.filterwarnings('ignore')
    # 设置关键参数 目标项目id 攻击配置文件数量 目标评分
    target_movie_id = 99
    attack_size = 200
    target_rate = 5

    timestamp = df_ratings[(df_ratings.movieId == target_movie_id)].sort_values(by='timestamp')['timestamp'].median()
    mean_rate = df_ratings["rating"].mean()
    # target rating
    last_user_id = df_ratings.loc[df_ratings['userId'].idxmax()].userId
    # 制作假配置文件
    for i in range(attack_size):
        timestamp += 1000
        new_row = {'userId': last_user_id + i,
                   'movieId': target_movie_id,
                   'rating': target_rate,
                   'timestamp': timestamp}
        df_ratings = df_ratings.append(new_row, ignore_index=True)
        # 获取随机电影id 避免攻击检测
        df_movies = pd.read_csv("./dataset/movies.csv", engine='python')
        df_random_movies = df_movies.sample(frac=0.001)
        movie_list = df_random_movies.Id.unique().tolist()
        for movie_id in movie_list:
            if movie_id == target_movie_id:
                continue
            new_row = {'userId': last_user_id + i,
                       'movieId': movie_id,
                       'rating': int(mean_rate),
                       'timestamp': timestamp}
            df_ratings = df_ratings.append(new_row, ignore_index=True)
            # df_ratings = pd.concat([df_ratings, new_row], ignore_index=True)
    # 获取假配置文件数据集
    return df_ratings
