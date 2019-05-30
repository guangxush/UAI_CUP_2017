# coding:utf-8
# 将原来的train文件增加了23维均值特征
import pandas as pd
import numpy as np
import os
import sys


def feature_process(in_file, out_file, data_path):
    data_flag = True if sys.argv[1] == 'test_public.csv' or sys.argv[1] == 'test_private.csv'else False
    train_dataframe = pd.read_csv(os.path.join(data_path, in_file), header=None)
    train_dataset = train_dataframe.values

    td_part1 = train_dataset[:, 0:24]  # 0-23维是POI~风力等级
    td_part1_mean = np.mean(td_part1, axis=0)
    td_part1_mean = np.tile(td_part1_mean, (len(td_part1), 1))
    td_part2 = td_part1-td_part1_mean  # 24-47维是POI~风力等级减去他们的均值
    if data_flag:
        td_part3 = train_dataset[:, 24:28]  # 48-52维是周几~终点ID
    else:
        td_part3 = train_dataset[:, 24:29]  # 48-52维是周几~订单量

    new_data = np.concatenate((td_part1, td_part2, td_part3), axis=1)
    format_data = pd.DataFrame(new_data)
    save_file_name = os.path.join(data_path, out_file)
    format_data.to_csv(save_file_name, index=False, header=False, float_format='%11.6f')


if __name__ == '__main__':
    in_file = sys.argv[1]  # train.csv 或者 dev.csv 或者test_public.csv 或者 test_private.csv
    out_file = sys.argv[2]  # train_53.csv 或者 dev_53.csv 或者test_public_52.csv 或者 test_private_52.csv
    feature_process(in_file, out_file, data_path='../data/')