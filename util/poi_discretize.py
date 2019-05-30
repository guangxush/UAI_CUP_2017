# coding:utf-8
# 对poi_clean中的数据进行离散化,以及统计7月份训练集中geo出现的次数
import numpy as np
import pandas as pd


def data_discretize():
    # 统计地点出现的热度,通过字典loc_items保存{地点，次数}
    fr1 = open('../raw_data/train_July.csv', 'r')
    records = []
    loc_items = {}
    for line in fr1:
        records.append(line.strip().split(','))
    for record in records:
        geo1 = record[-1]
        geo2 = record[-2]
        if geo1 in loc_items:
            loc_items[geo1] += 1
        else:
            loc_items[geo1] = 1
        if geo2 in loc_items:
            loc_items[geo2] += 1
        else:
            loc_items[geo2] = 1
    fr1.close()

    # 数据等宽分箱法：将数据均匀划分成n等份，每份的间距相等
    fr2 = open('../data/poi_clean.csv', 'r')
    fw = open('../data/poi_discretize.csv', 'w')
    fw.write('geo_id,geo_hot,poi1,poi2,poi3,poi4,poi5,poi6,poi7,poi8,poi9,poi10\n')
    geo_poi_record = []
    input_lines = fr2.readlines()[1:]
    for line in input_lines:
        geo_poi_record.append(line.strip().split(','))
    geo_poi_data = np.array(geo_poi_record)
    poi_data = geo_poi_data[:, 1:].astype('float32')
    m, n = np.shape(poi_data)  # 获取数据集行m列n（样本数和特征数)
    dis_mat = np.tile([0], np.shape(poi_data))  # 初始化离散化数据集
    for i in range(n-1):  # 遍历前n列特征列
        x = [l[i] for l in poi_data]  # 获取第i+1特征向量
        y = pd.cut(x, 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 调用cut函数，将特征离散化为10类
        for k in range(m):  # 将离散化值传入离散化数据集
            dis_mat[k][i] = y[k]
    i = 0
    for line in dis_mat[:]:
        fw.write(str(geo_poi_data[i][0]) + ',' + str(loc_items[geo_poi_data[i][0]]) + ',' +
                 (','.join(str(i) for i in line[:])) + '\n')
        i += 1
    fr2.close()
    fw.close()

if __name__ == '__main__':
    data_discretize()
