# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pandas.core.frame import DataFrame
import os
import numpy as np
import pandas as pd
import datetime
import time


# 不使用one-hot
def load_data_v1(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    train_dataset = train_dataframe.values

    x_train = train_dataset[:, 0:-2]
    y_train = train_dataset[:, -2].reshape(-1, 1).astype('float32')

    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    dev_dataset = dev_dataframe.values

    x_dev = dev_dataset[:, 0:-2]
    y_dev = dev_dataset[:, -2].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test = test_dataset

    print('X test shape:', x_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test


# 将离散型变量作One-Hot编码表示,mlp
def load_data_v2(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    train_dataset = train_dataframe.values

    x_train_con = train_dataset[:, 0:24].astype('float32')
    x_train_dis_time = train_dataset[:, 24:26].astype('float32')
    x_train_dis_loc = train_dataset[:, 26:28].astype('float32')

    one_hot_enc_time = OneHotEncoder(handle_unknown='ignore')
    x_train_dis_time = one_hot_enc_time.fit_transform(x_train_dis_time).toarray()
    one_hot_enc_loc = OneHotEncoder(handle_unknown='ignore')
    x_train_dis_loc = one_hot_enc_loc.fit_transform(x_train_dis_loc).toarray()

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_con = mm_scaler.fit_transform(x_train_con)

    x_train = np.concatenate((x_train_con, x_train_dis_time, x_train_dis_loc), axis=1)

    y_train = train_dataset[:, 28].reshape(-1, 1).astype('float32')
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    dev_dataset = dev_dataframe.values

    x_dev_con = dev_dataset[:, 0:24].astype('float32')
    x_dev_dis_time = dev_dataset[:, 24:26].astype('float32')
    x_dev_dis_loc = dev_dataset[:, 26:28].astype('float32')

    x_dev_dis_time = one_hot_enc_time.transform(x_dev_dis_time).toarray()
    x_dev_dis_loc = one_hot_enc_loc.transform(x_dev_dis_loc).toarray()

    x_dev_con = mm_scaler.transform(x_dev_con)

    x_dev = np.concatenate((x_dev_con, x_dev_dis_time, x_dev_dis_loc), axis=1)

    y_dev = dev_dataset[:, 28].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test_con = test_dataset[:, 0:24].astype('float32')
    x_test_dis_time = test_dataset[:, 24:26].astype('float32')
    x_test_dis_loc = test_dataset[:, 26:28].astype('float32')

    x_test_dis_time = one_hot_enc_time.transform(x_test_dis_time).toarray()
    x_test_dis_loc = one_hot_enc_loc.transform(x_test_dis_loc).toarray()

    x_test_con = mm_scaler.transform(x_test_con)

    x_test = np.concatenate((x_test_con, x_test_dis_time, x_test_dis_loc), axis=1)

    print('X test shape:', x_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test


# 全部样本的multi-task
def load_data_v3(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    train_dataset = train_dataframe.values

    dis_indices = [24, 25, 26, 27]

    one_hot_enc = OneHotEncoder(categorical_features=dis_indices, handle_unknown='ignore')
    x_train = one_hot_enc.fit_transform(train_dataset[:, 0:-2]).toarray()

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = mm_scaler.fit_transform(x_train)

    y_train_c = train_dataset[:, -1].reshape(-1, 1).astype('float32')
    y_train_r = train_dataset[:, -2].reshape(-1, 1).astype('float32')
    print('X train shape:', x_train.shape)
    print('y train classification shape:', y_train_c.shape)
    print('y train regression shape:', y_train_r.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    dev_dataset = dev_dataframe.values

    x_dev = one_hot_enc.transform(dev_dataset[:, 0:-2]).toarray()

    x_dev = mm_scaler.transform(x_dev)

    y_dev_c = dev_dataset[:, -1].reshape(-1, 1).astype('float32')
    y_dev_r = dev_dataset[:, -2].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev classification shape:', y_dev_c.shape)
    print('y dev regression shape:', y_dev_r.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test = one_hot_enc.transform(test_dataset).toarray()

    x_test = mm_scaler.transform(x_test)

    print('X test shape:', x_test.shape)

    return x_train, y_train_c, y_train_r, x_dev, y_dev_c, y_dev_r, x_test


# 0/1均衡的multi-task
def load_data_v4(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    train_dataset = train_dataframe.values

    dis_indices = [24, 25, 26, 27]

    one_hot_enc = OneHotEncoder(categorical_features=dis_indices, handle_unknown='ignore')
    x_train = one_hot_enc.fit_transform(train_dataset[:, 0:-2]).toarray()

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    mm_scaler.fit(x_train)

    train_dataset_high = train_dataframe[train_dataframe[29] == 1].values
    train_dataset_low = train_dataframe[train_dataframe[29] == 0].values
    high_shape = train_dataset_high.shape[0]
    low_shape = train_dataset_low.shape[0]
    print('High row count:', high_shape)
    print('Low row count:', low_shape)
    sample_indices = np.random.choice(low_shape, high_shape)

    train_dataset = np.concatenate((train_dataset_low[sample_indices], train_dataset_high))

    x_train = one_hot_enc.transform(train_dataset[:, 0:-2]).toarray()
    x_train = mm_scaler.transform(x_train)
    y_train_c = train_dataset[:, -1].reshape(-1, 1).astype('float32')
    y_train_r = train_dataset[:, -2].reshape(-1, 1).astype('float32')
    print('X train shape:', x_train.shape)
    print('y train classification shape:', y_train_c.shape)
    print('y train regression shape:', y_train_r.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    dev_dataset = dev_dataframe.values

    x_dev = one_hot_enc.transform(dev_dataset[:, 0:-2]).toarray()

    x_dev = mm_scaler.transform(x_dev)

    y_dev_c = dev_dataset[:, -1].reshape(-1, 1).astype('float32')
    y_dev_r = dev_dataset[:, -2].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev classification shape:', y_dev_c.shape)
    print('y dev regression shape:', y_dev_r.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test = one_hot_enc.transform(test_dataset).toarray()

    x_test = mm_scaler.transform(x_test)

    print('X test shape:', x_test.shape)

    return x_train, y_train_c, y_train_r, x_dev, y_dev_c, y_dev_r, x_test


# two-phase，使用全部train和dev做训练，实际训练时随机划分验证集合用于调参
def load_data_v5(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    train_dataframe = train_dataframe.append(dev_dataframe)
    train_dataset = train_dataframe.values

    dis_indices = [24, 25, 26, 27]

    one_hot_enc = OneHotEncoder(categorical_features=dis_indices, handle_unknown='ignore')
    x_train_all = one_hot_enc.fit_transform(train_dataset[:, 0:-2]).toarray()

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    mm_scaler.fit(x_train_all)

    train_dataset_high = train_dataframe[train_dataframe[29] == 1].values
    train_dataset_low = train_dataframe[train_dataframe[29] == 0].values
    high_shape = train_dataset_high.shape[0]
    low_shape = train_dataset_low.shape[0]
    print('High row count:', high_shape)
    print('Low row count:', low_shape)
    sample_indices = np.random.choice(low_shape, high_shape)

    train_dataset = np.concatenate((train_dataset_low[sample_indices], train_dataset_high))

    x_train_r = one_hot_enc.transform(train_dataset_high[:, 0:-2]).toarray()
    x_train_r = mm_scaler.transform(x_train_r)
    y_train_r = train_dataset_high[:, -2].reshape(-1, 1).astype('float32')

    x_train_c = one_hot_enc.transform(train_dataset[:, 0:-2]).toarray()
    x_train_c = mm_scaler.transform(x_train_c)
    y_train_c = train_dataset[:, -1].reshape(-1, 1).astype('float32')

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test = one_hot_enc.transform(test_dataset).toarray()

    x_test = mm_scaler.transform(x_test)

    x_train_c = np.concatenate((x_train_c,))

    print('X train classification shape:', x_train_c.shape)
    print('y train classification shape:', y_train_c.shape)
    print('X train regression shape:', x_train_r.shape)
    print('y train regression shape:', y_train_r.shape)
    print('X test shape:', x_test.shape)

    return x_train_c, y_train_c, x_train_r, y_train_r, x_test


def make_submission(result_path, results, model_name):
    submit_lines = ['test_id,count\n']
    for index, line in enumerate(results):
        submit_lines.append(','.join([str(index), str(line[0])]) + '\n')
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    result_file_name = 'result_' + model_name + '_' + timestamp + '.csv'
    with open(os.path.join(result_path, result_file_name), mode='w') as result_file:
        result_file.writelines(submit_lines)

# 去掉train中存在问题的数据，订单在1-9左右，并将离散型变量作One-Hot编码表示,mlp
def load_data_v6(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    train_dataframe = train_dataframe.loc[train_dataframe[28] < 10.0]
    train_dataset = train_dataframe.values


    x_train_con = train_dataset[:, 0:24].astype('float32')
    x_train_dis_time = train_dataset[:, 24:26].astype('float32')
    x_train_dis_loc = train_dataset[:, 26:28].astype('float32')

    one_hot_enc_time = OneHotEncoder(handle_unknown='ignore')
    x_train_dis_time = one_hot_enc_time.fit_transform(x_train_dis_time).toarray()
    one_hot_enc_loc = OneHotEncoder(handle_unknown='ignore')
    x_train_dis_loc = one_hot_enc_loc.fit_transform(x_train_dis_loc).toarray()

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_con = mm_scaler.fit_transform(x_train_con)

    x_train = np.concatenate((x_train_con, x_train_dis_time, x_train_dis_loc), axis=1)

    y_train = train_dataset[:, 28].reshape(-1, 1).astype('float32')
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    dev_dataset = dev_dataframe.values

    x_dev_con = dev_dataset[:, 0:24].astype('float32')
    x_dev_dis_time = dev_dataset[:, 24:26].astype('float32')
    x_dev_dis_loc = dev_dataset[:, 26:28].astype('float32')

    x_dev_dis_time = one_hot_enc_time.transform(x_dev_dis_time).toarray()
    x_dev_dis_loc = one_hot_enc_loc.transform(x_dev_dis_loc).toarray()

    x_dev_con = mm_scaler.transform(x_dev_con)

    x_dev = np.concatenate((x_dev_con, x_dev_dis_time, x_dev_dis_loc), axis=1)

    y_dev = dev_dataset[:, 28].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test_con = test_dataset[:, 0:24].astype('float32')
    x_test_dis_time = test_dataset[:, 24:26].astype('float32')
    x_test_dis_loc = test_dataset[:, 26:28].astype('float32')

    x_test_dis_time = one_hot_enc_time.transform(x_test_dis_time).toarray()
    x_test_dis_loc = one_hot_enc_loc.transform(x_test_dis_loc).toarray()

    x_test_con = mm_scaler.transform(x_test_con)

    x_test = np.concatenate((x_test_con, x_test_dis_time, x_test_dis_loc), axis=1)

    print('X test shape:', x_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test

# 去掉train、dev中存在问题的数据，订单在1-9左右，每个区间随机选取9000条数据
def load_data_v7(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    train_dataframe = train_dataframe.loc[train_dataframe[28] < 10.0]
    grouped = train_dataframe.groupby(train_dataframe[28])
    print("数据集中需求量的分布情况：" +str(grouped.size()))  #每一组的数据量大小
    # 从数据集需求量1-9中每一个随机挑选9000条数据
    data_1 = train_dataframe.loc[train_dataframe[28] == 1]
    data_valid = data_1.take(np.random.permutation(len(data_1))[:9000])

    for i in range(2, 10):
        temp = train_dataframe.loc[train_dataframe[28] == i]
        data_valid = data_valid.append(temp.take(np.random.permutation(len(temp))[:9000]))

    train_dataframe2 = DataFrame(data_valid)
    train_dataset = train_dataframe2.values

    x_train_con = train_dataset[:, 0:24].astype('float32')
    x_train_dis_time = train_dataset[:, 24:26].astype('float32')
    x_train_dis_loc = train_dataset[:, 26:28].astype('float32')

    one_hot_enc_time = OneHotEncoder(handle_unknown='ignore')
    x_train_dis_time = one_hot_enc_time.fit_transform(x_train_dis_time).toarray()
    one_hot_enc_loc = OneHotEncoder(handle_unknown='ignore')
    x_train_dis_loc = one_hot_enc_loc.fit_transform(x_train_dis_loc).toarray()

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_con = mm_scaler.fit_transform(x_train_con)

    x_train = np.concatenate((x_train_con, x_train_dis_time, x_train_dis_loc), axis=1)

    y_train = train_dataset[:, 28].reshape(-1, 1).astype('float32')
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    dev_dataframe = dev_dataframe.loc[dev_dataframe[28] < 10.0]
    grouped = dev_dataframe.groupby(dev_dataframe[28])
    print("数据集中需求量的分布情况："+str(grouped.size()))  # 每一组的数据量大小
    # 从数据集需求量1-9中每一个随机挑选9000条数据
    data_1 = dev_dataframe.loc[dev_dataframe[28] == 1]
    data_valid = data_1.take(np.random.permutation(len(data_1))[:9000])

    for i in range(2, 10):
        temp = dev_dataframe.loc[dev_dataframe[28] == i]
        data_valid = data_valid.append(temp.take(np.random.permutation(len(temp))[:9000]))

    dev_dataframe2 = DataFrame(data_valid)
    dev_dataset = dev_dataframe2.values

    x_dev_con = dev_dataset[:, 0:24].astype('float32')
    x_dev_dis_time = dev_dataset[:, 24:26].astype('float32')
    x_dev_dis_loc = dev_dataset[:, 26:28].astype('float32')

    x_dev_dis_time = one_hot_enc_time.transform(x_dev_dis_time).toarray()
    x_dev_dis_loc = one_hot_enc_loc.transform(x_dev_dis_loc).toarray()

    x_dev_con = mm_scaler.transform(x_dev_con)

    x_dev = np.concatenate((x_dev_con, x_dev_dis_time, x_dev_dis_loc), axis=1)

    y_dev = dev_dataset[:, 28].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test_con = test_dataset[:, 0:24].astype('float32')
    x_test_dis_time = test_dataset[:, 24:26].astype('float32')
    x_test_dis_loc = test_dataset[:, 26:28].astype('float32')

    x_test_dis_time = one_hot_enc_time.transform(x_test_dis_time).toarray()
    x_test_dis_loc = one_hot_enc_loc.transform(x_test_dis_loc).toarray()

    x_test_con = mm_scaler.transform(x_test_con)

    x_test = np.concatenate((x_test_con, x_test_dis_time, x_test_dis_loc), axis=1)

    print('X test shape:', x_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test


# 不使用POI数据，将离散型变量作One-Hot编码表示,mlp
def load_data_v8(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    train_dataset = train_dataframe.values

    x_train_con = train_dataset[:, 20:24].astype('float32')
    x_train_dis_time = train_dataset[:, 24:26].astype('float32')
    x_train_dis_loc = train_dataset[:, 26:28].astype('float32')

    one_hot_enc_time = OneHotEncoder(handle_unknown='ignore')
    x_train_dis_time = one_hot_enc_time.fit_transform(x_train_dis_time).toarray()
    one_hot_enc_loc = OneHotEncoder(handle_unknown='ignore')
    x_train_dis_loc = one_hot_enc_loc.fit_transform(x_train_dis_loc).toarray()

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_con = mm_scaler.fit_transform(x_train_con)

    x_train = np.concatenate((x_train_con, x_train_dis_time, x_train_dis_loc), axis=1)

    y_train = train_dataset[:, 28].reshape(-1, 1).astype('float32')
    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'), header=None)
    dev_dataset = dev_dataframe.values

    x_dev_con = dev_dataset[:, 20:24].astype('float32')
    x_dev_dis_time = dev_dataset[:, 24:26].astype('float32')
    x_dev_dis_loc = dev_dataset[:, 26:28].astype('float32')

    x_dev_dis_time = one_hot_enc_time.transform(x_dev_dis_time).toarray()
    x_dev_dis_loc = one_hot_enc_loc.transform(x_dev_dis_loc).toarray()

    x_dev_con = mm_scaler.transform(x_dev_con)

    x_dev = np.concatenate((x_dev_con, x_dev_dis_time, x_dev_dis_loc), axis=1)

    y_dev = dev_dataset[:, 28].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test_con = test_dataset[:, 20:24].astype('float32')
    x_test_dis_time = test_dataset[:, 24:26].astype('float32')
    x_test_dis_loc = test_dataset[:, 26:28].astype('float32')

    x_test_dis_time = one_hot_enc_time.transform(x_test_dis_time).toarray()
    x_test_dis_loc = one_hot_enc_loc.transform(x_test_dis_loc).toarray()

    x_test_con = mm_scaler.transform(x_test_con)

    x_test = np.concatenate((x_test_con, x_test_dis_time, x_test_dis_loc), axis=1)

    print('X test shape:', x_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test


# 不使用one-hot,数据加了23维均值
def load_data_v9(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train_53.csv'), header=None)
    train_dataset = train_dataframe.values

    x_train = train_dataset[:, 0:-1]
    y_train = train_dataset[:, -1].reshape(-1, 1).astype('float32')

    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)

    dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev_53.csv'), header=None)
    dev_dataset = dev_dataframe.values

    x_dev = dev_dataset[:, 0:-1]
    y_dev = dev_dataset[:, -1].reshape(-1, 1).astype('float32')

    print('X dev shape:', x_dev.shape)
    print('y dev shape:', y_dev.shape)

    test_dataframe = pd.read_csv(os.path.join(data_path, 'test_public_52.csv'), header=None)
    test_dataset = test_dataframe.values.astype('float32')

    x_test = test_dataset

    print('X test shape:', x_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test


if __name__ == '__main__':
    load_data_v9(data_path='../data/')
