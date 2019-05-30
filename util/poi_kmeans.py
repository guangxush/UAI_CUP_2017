#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans(input_file, output_file):
    fw = open(output_file, 'w')
    fw.write('geo_id,poi1count,poi2count,poi3count,poi4count,poi5count,poi6count,poi6count,poi7count,poi8count,poi9count,poi10count\n')
    poidata = pd.read_csv(input_file, header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    poidata = poidata[1:].astype('float32')
    dataSet = poidata.values

    k = 10
    clf = KMeans(n_clusters=k)  # 设定k调用KMeans算法
    s = clf.fit(dataSet)  # 加载数据集合
    numSamples = len(dataSet)
    labels = clf.labels_
    '''mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 画出所有样例点 属于同一分类的绘制同样的颜色
    for i in range(numSamples):
        plt.plot(dataSet[i][0], dataSet[i][1], dataSet[i][2], dataSet[i][3],dataSet[i][4], dataSet[i][5], dataSet[i][6], dataSet[i][7],dataSet[i][8], dataSet[i][9],mark[clf.labels_[i]], markersize=4)# mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画出质心，用特殊图型
    centroids = clf.cluster_centers_
    for i in range(k):
        plt.plot(centroids[i][0], centroids[i][1], centroids[i][2], centroids[i][3],centroids[i][4], centroids[i][5], centroids[i][6], centroids[i][7],centroids[i][8], centroids[i][9],mark[i], markersize=6)
        #print (centroids[i][0], centroids[i][1], centroids[i][2], centroids[i][3],centroids[i][4], centroids[i][5], centroids[i][6], centroids[i][7],centroids[i][8], centroids[i][9])
    plt.show()'''
    result = clf.predict(dataSet)
    poigeo_data = pd.read_csv(input_file, header=None, usecols=[0])
    poigeo = poigeo_data[1:].values
    result_fina = []
    for i in range(len(result)):
        temp = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        temp[result[i]] = '1'
        result_fina.append(temp)
    output_lines = []
    for i in range(len(result)):
        temp_line = poigeo[i][0]+','+','.join(result_fina[i])+'\n'
        output_lines.append(temp_line)
    fw.writelines(output_lines)
    fw.close()

if __name__ == '__main__':
    kmeans(input_file='../data/poi_clean.csv', output_file='../data/poi_kmeans.csv')