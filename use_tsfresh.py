#　tsfreshで時系列特徴量を抽出するプログラム

from math import e
import matplotlib.pylab as plt
from numpy.core.arrayprint import printoptions
from tsfresh import extract_features, extract_relevant_features, feature_extraction, select_features
from tsfresh.feature_extraction.settings import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA


c_m_count = 24  # 各状態行動の回数　今回は24回ずつ
c_count = 4  # 状態数
m_count = 3  # 行動数
user_name = str(sys.argv[1])  # 被験者名
sensor_num = 30  # 服に装着したセンサの数　今回は30

# 加速度，角速度整形用引数　整数型に変形しているが文字
acc_gyr_changer = str(sys.argv[2])

# 保存先指定　文字列結合はjoinの方がいいかも
save_path = "/Users/matsuokoki/Desktop/arm_move_data_" + \
    user_name + "/" + user_name + "_feature_data"


def make_pandas_dataframe(user_name, condition_count, move_count, move_condition_count):
    move_dir_path = "/Users/matsuokoki/Desktop/arm_move_data_" + user_name  # 各人物別にデータを取り出す

    path = move_dir_path + "/" + user_name + "_condition" + \
        str(condition_count) + "_move" + str(move_count)

    dir_path = path + "/output_" + user_name + "_condition" + \
        str(condition_count) + "_move" + str(move_count) + "_count" + \
        str(move_condition_count)

    os.chdir(dir_path)
    npy_file_list = os.listdir()

    npy_time_file = npy_file_list[1]
    npy_file = npy_file_list[2]  # imu.npyのこと、ちゃんと名前で指定しよう
    array1 = np.load(npy_file)
    array2 = np.load(npy_time_file)
    tmp_time_array = np.zeros(array2.shape)
    time = 0

    # 時間が実行時点を0として記録されているため差分を取り直した
    for i in range(len(array2)):
        if i == 0:
            np.put(tmp_time_array, i, time)
        else:

            tmp_time = array2[i] - array2[i-1]
            time += tmp_time
            np.put(tmp_time_array, i, time)

    # 以下dataframe作成用に整形
    result = []
    for k in range(len(array1)):
        time_us = tmp_time_array[k]
        for j in range(len(array1[k, :])):
            result.append(np.insert(array1[k, j], 0, [time_us, j+1]))
    array1 = np.array(result)

    # ここまで
    df = pd.DataFrame(array1, columns=[
        "time_us", "sensor_id", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])

    if acc_gyr_changer == "acc":
        # 角速度削除用
        df = df.drop("gyr_x", axis=1)
        df = df.drop("gyr_y", axis=1)
        df = df.drop("gyr_z", axis=1)
    elif acc_gyr_changer == "gyr":
        # 加速度削除用
        df = df.drop("acc_x", axis=1)
        df = df.drop("acc_y", axis=1)
        df = df.drop("acc_z", axis=1)

    print(df)
    print(df.shape)

    return df

# 動作ラベル作成用関数　0~11で動作1~12を管理


def make_condition_move_label(condition_counter, move_counter):
    if condition_counter == 1:
        if move_counter == 1:
            return 1
        elif move_counter == 2:
            return 2
        elif move_counter == 3:
            return 3
        else:
            sys.exit('Error!')
    elif condition_counter == 2:
        if move_counter == 1:
            return 4
        elif move_counter == 2:
            return 5
        elif move_counter == 3:
            return 6
        else:
            sys.exit('Error!')
    elif condition_counter == 3:
        if move_counter == 1:
            return 7
        elif move_counter == 2:
            return 8
        elif move_counter == 3:
            return 9
        else:
            sys.exit('Error!')
    elif condition_counter == 4:
        if move_counter == 1:
            return 10
        elif move_counter == 2:
            return 11
        elif move_counter == 3:
            return 12
        else:
            sys.exit('Error!')

    else:
        sys.exit('Error!')


def main():
    counter = 0
    label_arr = np.zeros(c_count*m_count*c_m_count, dtype=np.int)
    data_arr = np.zeros((30, 2367))
    for j in range(c_count):
        for k in range(m_count):
            label_num = make_condition_move_label(j+1, k+1)
            label = np.full(sensor_num, label_num)
            for l in range(c_m_count):
                df = make_pandas_dataframe(user_name, j+1, k+1, l+1)

                np.put(label_arr, counter, label_num)
                print(label_arr)

                features = extract_features(
                    df, column_id="sensor_id", column_sort="time_us", column_kind=None, column_value=None)

                #　特徴量抽出
                # print(features_filtered_direct)
                print(features)
                print(features.shape)
                print(data_arr)

                if counter == 0:
                    data_arr = features

                elif counter == 1:
                    data_arr = np.stack([data_arr, features])

                elif counter > 1:
                    data_arr = np.block([[[data_arr]], [[features]]])
                else:
                    sys.exit('Error!')
                print(data_arr.shape)
                counter += 1

    print(data_arr.shape)
    os.chdir(save_path)
    if acc_gyr_changer == "acc":
        new_save_path = save_path + "acc_"
    elif acc_gyr_changer == "gyr":
        new_save_path = save_path + "gyr_"
    else:
        new_save_path = save_path + "accgyr_"

    data_path = new_save_path + "feature.npy"
    np.save(data_path, data_arr)
    label_path = new_save_path + "label.npy"
    np.save(label_path, label_arr)


if __name__ == "__main__":
    main()
