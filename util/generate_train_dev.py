# coding:utf-8
import sys
import time


# 根据日期查询周几,输入：日期(2017-08-01),输出：周几(1-7)
def query_week(date_str):
    if int(time.strftime("%w", time.strptime(date_str, "%Y-%m-%d"))) == 0:
        return 7
    return int(time.strftime("%w", time.strptime(date_str, "%Y-%m-%d")))


def generate_train(input_file, output_file):
    # 读取poi数据,将其转换为字典(key:geo_id,value:poiid1-10)
    fr1 = open('../data/poi_clean.csv', 'r')
    poi = {}
    for line in fr1.readlines()[1:]:
        line = line.strip().split(',')
        geo_id = line[0]
        poi_value = line[1:]
        poi[geo_id] = poi_value
    fr1.close()

    # 读取天气信息将其转换成字典(key:day,value:(key:hour,value:weather_condition))
    fr2 = open('../data/weather_clean.csv', 'r')
    raw_wea_record = []
    for line in fr2:
        raw_wea_record.append(line.strip().split(','))
    weather_record = {}
    for record in raw_wea_record:
        day_hour = record[0] + ',' + record[1]
        weather_record[day_hour] = record[2:]
    fr2.close()

    # 生成POI_ID -> int_id 的字典（提供了1234个POI点，因此ID为1-1234，0用于表示未知点）
    fr3 = open('../data/poi_clean.csv', 'r')
    poi_id = {}
    for index, line in enumerate(fr3.readlines()[1:]):
        line = line.strip().split(',')
        geo_id = line[0]
        poi_id[geo_id] = index + 1
    fr3.close()

    fr = open(input_file, 'r')
    fw = open(output_file, 'w')

    output_lines = []
    input_lines = fr.readlines()[1:]

    for input_line in input_lines:
        line = input_line.strip().split(',')
        start_poi = poi.get(line[1], ['1'] * 10)
        end_poi = poi.get(line[2], ['1'] * 10)
        weather_condition = weather_record.get((line[3] + ',' + line[4]))
        day_of_the_week = str(query_week(line[3]))
        create_hour = line[4]
        start_point = str(poi_id.get(line[1], 0))
        end_point = str(poi_id.get(line[2], 0))
        order_count = line[5]

        temp_line = ','.join(start_poi + end_poi + weather_condition +
                             [day_of_the_week, create_hour, start_point, end_point, order_count]) + '\n'
        output_lines.append(temp_line)

    fw.writelines(output_lines)
    fw.close()
    fr.close()


if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    generate_train(in_file, out_file)
