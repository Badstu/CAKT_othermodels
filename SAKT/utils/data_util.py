import csv
import copy
import numpy as np


def get_rows_from_file(file_name):
    rows = []
    with open(file_name, "r") as data_file:
        reader = csv.reader(data_file, delimiter=',')
        for row in reader:
            rows.append(row)
        data_file.close()
    return rows


def read_data_from_csv_file(train_file, test_file, min_len, max_sequence_len):
    print("+" * 10 + "reading data" + "+" * 10)

    # rows 所有记录
    rows = []
    # 所有知识点的数量
    max_skills = 0

    tuple_rows = []
    train_rows = []
    rows += get_rows_from_file(train_file)

    train_num = int(len(rows))
    print(f"train record num: {train_num}")

    rows += get_rows_from_file(test_file)

    index = 0
    while index < len(rows) - 1:
        problems_num = int(len(rows[index + 1]))

        # 对最小的长度进行截断
        if problems_num <= min_len:
            # 跳过当前三行记录
            index += 3
            continue

        tmp_max_skill = max(map(int, rows[index + 1]))
        if tmp_max_skill > max_skills:
            max_skills = tmp_max_skill

        if problems_num < max_sequence_len:
            problems = np.zeros(max_sequence_len)
            correct = np.zeros(max_sequence_len)
            for j in range(problems_num):
                problems[max_sequence_len - j - 1] = problems[max_sequence_len - j - 1] + int(
                    rows[index + 1][problems_num - j - 1])
                correct[max_sequence_len - j - 1] = correct[max_sequence_len - j - 1] + int(
                    rows[index + 2][problems_num - j - 1])
                tup = (problems_num, problems, correct)
            tuple_rows.append(tup)
        # #
        else:
            start_idx = 0
            while max_sequence_len + start_idx <= problems_num:
                tup = (max_sequence_len, [int(i) for i in rows[index + 1][start_idx:max_sequence_len + start_idx]],
                       [int(i) for i in rows[index + 2][start_idx:max_sequence_len + start_idx]])
                start_idx += max_sequence_len
                tuple_rows.append(tup)
            # test_rows.append((rows[index], rows[index+1][-max_num_problems:], rows[index+2][-max_num_problems:]))

        index += 3
        if index == train_rows:
            train_rows = copy.deepcopy(tuple_rows)
            tuple_rows = []

    test_rows = copy.deepcopy(tuple_rows)
    # print "The number of students is ", len(test_rows)
    # print "The number of train students is ", len(train_rows)
    # print "Finish reading data"
    return train_rows, test_rows, max_sequence_len, max_skills + 1
