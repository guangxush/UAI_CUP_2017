# coding:utf-8
# weather.csv --> trainWeather.csv 原始数据中找出day,hour,weathercode
import time


def weather_data_process(fname):
    fr = open(fname, 'r')
    fw = open('../data/weather_clean.csv', 'w')
    fw.write('day,hour,weather_code,feels_like,visibility,wind_scale\n')
    raw_weathers = []
    for line in fr:
        raw_weathers.append(line.strip().split(','))

    wea_record = []
    for raw_weather in raw_weathers:
        one_record = [raw_weather[0], raw_weather[2], raw_weather[4], raw_weather[7], raw_weather[-1]]
        wea_record.append(one_record)

    weather_items = {}
    for i in range(1, len(wea_record))[::-1]:
        record = wea_record[i]
        day = time.strftime("%Y-%m-%d", time.strptime(str(record[0].split()[0]), '%Y-%m-%d'))  # 日期
        hour = (record[0].split()[1].split(':')[0])  # 小时
        day_hour = day + ',' + hour
        # minute = int(record[0].split()[1].split(':')[1])  # 分钟,暂时用不到
        weather_items[day_hour] = ','.join([str(record[1]), str(record[2]), str(record[3]), str(record[4])])

    for items in weather_items:
        fw.write(items + ',' + weather_items[items] + '\n')
    fr.close()
    fw.close()


if __name__ == '__main__':
    weather_data_process('../raw_data/weather.csv')
