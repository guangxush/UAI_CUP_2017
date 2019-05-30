# coding:utf-8
# poi.csv --> poiValid.csv 原始数据中找出geo_id,poi1count,poi2count,...poi10count


def poi_data_process(fname):
    fr = open(fname, 'r')
    fw = open('../data/poi_clean.csv', 'w')
    fw.write('geo_id,poi1count,poi2count,poi3count,poi4count,poi5count,poi6count,poi6count,poi7count,poi8count,poi9count,poi10count\n')
    record = []
    for line in fr:
        record.append(line.strip().split(','))
    for i in range(0, len(record)):
        temp = record[i][0:21:2]
        for j in range(0, len(temp)-1):
            fw.write(str(temp[j]) + ",")
        fw.write(temp[-1] + "\n")
    fr.close()
    fw.close()

if __name__ == '__main__':
    poi_data_process('../raw_data/poi.csv')
