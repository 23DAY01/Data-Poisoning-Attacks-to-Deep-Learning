import pandas as pd
import seaborn as sns
import scipy.stats as stats
from attack_detect.models import Rate, SuspiciousUser
from attack_detect.utils import optimum_chunk_size, get_sample_entropy_list, get_sample_average_list, \
    z_score_entropy_plot, get_optimal_chunk_size
from data_poison_attack_Normal.attack import attack


def check_shilling_attack():
    # 读取数据集
    df_ratings = pd.read_csv("../dataset/ratings.csv", engine='python')
    attack_size = 100
    target_movie_id = 70
    df_target_movie = df_ratings[(df_ratings.MovieID == target_movie_id)]
    # 获取假配置文件数据集
    new_df = attack
    sns.countplot(x='Rating', data=df_target_movie)
    df_attacked_movie = new_df[(new_df.MovieID == target_movie_id)].sort_values(by='Timestamp')
    df_target_movie['Rating'].mean()
    sns.countplot(x='Rating', data=df_attacked_movie)
    movies = Rate.objects.filter(movie_id=target_movie_id).all()
    df = pd.DataFrame(list(movies.values()))
    df.rename(columns={'user_id': 'UserID', 'movie_id': 'MovieID', 'rating': 'Rating', 'timestamp': 'Timestamp'},
              inplace=True)
    df = df.drop('id', 1)
    df_attacked_movie = df[(df.MovieID == target_movie_id)].sort_values(by='Timestamp')
    suspicious_user_ids = []
    threshold = 2.5
    chunk_size = 20
    if len(df_attacked_movie) > chunk_size:
        # 获取最佳攻击数量
        chunk_size = get_optimal_chunk_size(attack_size)
        # 获取样本条目列表
        sample_entropy_list, entropy_chunk_list = get_sample_entropy_list(df_attacked_movie, chunk_size=chunk_size)
        # 获取样本平均值列表
        sample_average_list, average_chunk_list = get_sample_average_list(df_attacked_movie, chunk_size=chunk_size)
        z_score_sample_entropy_list = stats.zscore(sample_entropy_list)
        z_score_sample_average_list = stats.zscore(sample_average_list)
        entropy_peak_indexes = [i for i, v in enumerate(z_score_sample_entropy_list) if abs(v) > threshold]
        average_peak_indexes = [i for i, v in enumerate(z_score_sample_average_list) if abs(v) > threshold]
        final_attack_indexes = list(set(entropy_peak_indexes).intersection(average_peak_indexes))
        for i in final_attack_indexes:
            df_chunk = entropy_chunk_list[i]
            suspicious_user_ids.extend(df_chunk['UserID'].tolist())
        print(suspicious_user_ids)
        for user_id in suspicious_user_ids:
            SuspiciousUser.objects.get_or_create(user_id=user_id)
