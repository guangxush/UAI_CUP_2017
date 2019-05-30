# coding:utf-8
import sys


def generate_train_input(input_file, output_file):

    fr = open(input_file, 'r')
    fw = open(output_file, 'w')
    raw_records = []
    records = []
    count_items = {}
    train_lines = []

    for line in fr:
        raw_records.append(line.strip().split(','))

    for raw_record in raw_records[1:]:
        one_record = [raw_record[-2], raw_record[-1], raw_record[3], raw_record[4], raw_record[5]]
        records.append(one_record)

    for record in records:
        count_key = (record[0], record[1], record[2], record[3])
        count_items[count_key] = count_items.get(count_key, 0) + 1

    train_lines.append('id,start_geo_id,end_geo_id,create_date,create_hour,order_count\n')
    for idx, count_item in enumerate(count_items.items()):
        temp_line = ','.join([str(idx), count_item[0][0], count_item[0][1], count_item[0][2],
                              str(int(count_item[0][3])), str(count_item[1])])
        train_lines.append(temp_line + '\n')
    fw.writelines(train_lines)

    fw.close()
    fr.close()

if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    generate_train_input(in_file, out_file)
