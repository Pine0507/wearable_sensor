import numpy as np
import pandas as pd
import copy
import optuna
import itertools
import lightgbm as lgb
import time
import sys
import csv
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tsfresh.utilities.dataframe_functions import impute

load_data_acc_gyr = str(sys.argv[1])  # acc or gyr or accgyr

# 読み出し元パス　要変更

tsukamoto_path = "C:\\Users\\matsu\\OneDrive\\デスクトップ\\arm_data\\arm_move_data_tsukamoto\\tsukamoto_feature_" + load_data_acc_gyr
iwata_path = "C:\\Users\\matsu\\OneDrive\\デスクトップ\\arm_data\\arm_move_data_iwata\\iwata_feature_" + \
    load_data_acc_gyr
matsuo_path = "C:\\Users\\matsu\\OneDrive\\デスクトップ\\arm_data\\arm_move_data_matsuo\\matsuo_feature_" + \
    load_data_acc_gyr
fuji_path = "C:\\Users\\matsu\\OneDrive\\デスクトップ\\arm_data\\arm_move_data_fuji\\fuji_feature_" + load_data_acc_gyr
yamagami_path = "C:\\Users\\matsu\\OneDrive\\デスクトップ\\arm_data\\arm_move_data_yamagami\\yamagami_feature_" + load_data_acc_gyr

tsukamoto_label_array = np.load(
    tsukamoto_path + "\\tsukamoto_feature_datalabel.npy")
iwata_label_array = np.load(iwata_path + "\\iwata_feature_datalabel.npy")
matsuo_label_array = np.load(matsuo_path + "\\matsuo_feature_datalabel.npy")
fuji_label_array = np.load(fuji_path + "\\fuji_feature_datalabel.npy")
yamagami_label_array = np.load(
    yamagami_path + "\\yamagami_feature_datalabel.npy")

write_path = "C:/Users/matsu/OneDrive/デスクトップ/report_lightgbm_tune10_f1" + \
    load_data_acc_gyr + ".csv"


average = "macro"


#　センサ数別に配列を
def one_array_extract(array, array_index):
    return_array = array[:, array_index, :]

    return return_array


def two_array_extract(array, array_index_one, array_index_two):
    array_one = one_array_extract(array, array_index_one)
    array_two = one_array_extract(array, array_index_two)
    return_array = np.stack([array_one, array_two], 1)

    return return_array


# 交差検証用の配列生成
def return_train_cross_valid(array1, array2, array3, array4, j):
    if j == 0:
        return_array = np.concatenate([array1, array2, array3])
    elif j == 1:
        return_array = np.concatenate([array1, array2, array4])
    elif j == 2:
        return_array = np.concatenate([array1, array3, array4])
    elif j == 3:
        return_array = np.concatenate([array2, array3, array4])
    else:
        sys.exit()

    return return_array


def return_valid_cross_valid(array1, array2, array3, array4, j):
    if j == 0:
        return_array = array4
    elif j == 1:
        return_array = array3
    elif j == 2:
        return_array = array2
    elif j == 3:
        return_array = array1
    else:
        sys.exit()

    return return_array


#　optuna適用関数　引数が多すぎるため構造体にすべき
def bayes_objective(trial, tsukamoto_feature_data, matsuo_feature_data, yamagami_feature_data, fuji_feature_data, iwata_feature_data,
                    tsukamoto_label_data, matsuo_label_data, yamagami_label_data, fuji_label_data, iwata_label_data, i):

    params = {

        'objective': 'multiclass',
        'metric': 'multi_error',
        'boosting_type': 'gbdt',
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'feature_fraction': 0.9,
        'learning_rate': 0.1,
        'num_iteration': 100,
        # 調整パラメータ
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 60),

        'force_col_wise': True,



        'num_class': 12,
        'n_jobs': -1,  # cpu使用コア設定,-1で全コア使用
        'early_stopping_round': 5,
        'seed': 0
    }

    fi_score_array = np.empty(4)

    #　各学習データパターン別に4人での交差検証　
    if i == 0:  # test:tsukamoto

        for j in range(4):
            train_array = return_train_cross_valid(
                matsuo_feature_data, yamagami_feature_data, fuji_feature_data, iwata_feature_data, j)
            valid_array = return_valid_cross_valid(
                matsuo_feature_data, yamagami_feature_data, fuji_feature_data, iwata_feature_data, j)
            train_label = return_train_cross_valid(
                matsuo_label_data, yamagami_label_data, fuji_label_data, iwata_label_data, j)
            valid_label = return_valid_cross_valid(
                matsuo_label_data, yamagami_label_data, fuji_label_data, iwata_label_data, j)

            lgb_train = lgb.Dataset(
                train_array, train_label, free_raw_data=False)
            lgb_eval = lgb.Dataset(
                valid_array, valid_label, free_raw_data=False)
            model = lgb.train(params, lgb_train, valid_sets=lgb_eval,
                              num_boost_round=100)
            preds = model.predict(valid_array)

            pred_labels = np.argmax(preds, axis=1)

            f1_sco = f1_score(valid_label, pred_labels, average=average)
            np.put(fi_score_array, j, f1_sco)
        return_f1 = np.average(fi_score_array)

    elif i == 1:  # test:iwata

        for j in range(4):
            train_array = return_train_cross_valid(
                matsuo_feature_data, yamagami_feature_data, fuji_feature_data, tsukamoto_feature_data, j)
            valid_array = return_valid_cross_valid(
                matsuo_feature_data, yamagami_feature_data, fuji_feature_data, tsukamoto_feature_data, j)
            train_label = return_train_cross_valid(
                matsuo_label_data, yamagami_label_data, fuji_label_data, tsukamoto_label_data, j)
            valid_label = return_valid_cross_valid(
                matsuo_label_data, yamagami_label_data, fuji_label_data, tsukamoto_label_data, j)
            lgb_train = lgb.Dataset(
                train_array, train_label, free_raw_data=False)
            lgb_eval = lgb.Dataset(
                valid_array, valid_label, free_raw_data=False)
            model = lgb.train(params, lgb_train, valid_sets=lgb_eval,
                              num_boost_round=100)
            preds = model.predict(valid_array)

            pred_labels = np.argmax(preds, axis=1)

            f1_sco = f1_score(valid_label, pred_labels, average=average)
            np.put(fi_score_array, j, f1_sco)
        return_f1 = np.average(fi_score_array)

    elif i == 2:  # test:fuji

        for j in range(4):
            train_array = return_train_cross_valid(
                matsuo_feature_data, yamagami_feature_data, iwata_feature_data, tsukamoto_feature_data, j)
            valid_array = return_valid_cross_valid(
                matsuo_feature_data, yamagami_feature_data, iwata_feature_data, tsukamoto_feature_data, j)
            train_label = return_train_cross_valid(
                matsuo_label_data, yamagami_label_data, iwata_label_data, tsukamoto_label_data, j)
            valid_label = return_valid_cross_valid(
                matsuo_label_data, yamagami_label_data, iwata_label_data, tsukamoto_label_data, j)
            lgb_train = lgb.Dataset(
                train_array, train_label, free_raw_data=False)
            lgb_eval = lgb.Dataset(
                valid_array, valid_label, free_raw_data=False)
            model = lgb.train(params, lgb_train, valid_sets=lgb_eval,
                              num_boost_round=100)
            preds = model.predict(valid_array)

            pred_labels = np.argmax(preds, axis=1)

            f1_sco = f1_score(valid_label, pred_labels, average=average)
            np.put(fi_score_array, j, f1_sco)
        return_f1 = np.average(fi_score_array)

    elif i == 3:  # test:yamagami

        for j in range(4):
            train_array = return_train_cross_valid(
                matsuo_feature_data, fuji_feature_data, tsukamoto_feature_data, iwata_feature_data, j)
            valid_array = return_valid_cross_valid(
                matsuo_feature_data, fuji_feature_data, tsukamoto_feature_data, iwata_feature_data, j)
            train_label = return_train_cross_valid(
                matsuo_label_data, fuji_label_data, tsukamoto_label_data, iwata_label_data, j)
            valid_label = return_valid_cross_valid(
                matsuo_label_data, fuji_label_data, tsukamoto_label_data, iwata_label_data, j)

            lgb_train = lgb.Dataset(
                train_array, train_label, free_raw_data=False)
            lgb_eval = lgb.Dataset(
                valid_array, valid_label, free_raw_data=False)
            model = lgb.train(params, lgb_train, valid_sets=lgb_eval,
                              num_boost_round=100)
            preds = model.predict(valid_array)

            pred_labels = np.argmax(preds, axis=1)

            f1_sco = f1_score(valid_label, pred_labels, average=average)
            np.put(fi_score_array, j, f1_sco)
        return_f1 = np.average(fi_score_array)

    elif i == 4:  # test:matsuo

        for j in range(4):
            train_array = return_train_cross_valid(
                tsukamoto_feature_data, yamagami_feature_data, fuji_feature_data, iwata_feature_data, j)
            valid_array = return_valid_cross_valid(
                tsukamoto_feature_data, yamagami_feature_data, fuji_feature_data, iwata_feature_data, j)
            train_label = return_train_cross_valid(
                tsukamoto_label_data, yamagami_label_data, fuji_label_data, iwata_label_data, j)
            valid_label = return_valid_cross_valid(
                tsukamoto_label_data, yamagami_label_data, fuji_label_data, iwata_label_data, j)

            lgb_train = lgb.Dataset(
                train_array, train_label, free_raw_data=False)
            lgb_eval = lgb.Dataset(
                valid_array, valid_label, free_raw_data=False)
            model = lgb.train(params, lgb_train, valid_sets=lgb_eval,
                              num_boost_round=100)
            preds = model.predict(valid_array)

            pred_labels = np.argmax(preds, axis=1)

            f1_sco = f1_score(valid_label, pred_labels, average=average)
            np.put(fi_score_array, j, f1_sco)
        return_f1 = np.average(fi_score_array)
    else:
        sys.exit()

    return return_f1


def use_lightgbm_model(tsukamoto_feature_array, matsuo_feature_array, yamagami_feature_array, fuji_feature_array, iwata_feature_array,
                       tsukamoto_label_data, matsuo_label_data, yamagami_label_data, fuji_label_data, iwata_label_data, use_sensor_num, use_sensor_num1, use_sensor_num2):

    #　値の補完処理　いちいちやってると処理が遅くなるのでtsfreshで抽出する時一緒にやるべき
    if use_sensor_num > 1:
        # print(X_train.shape[1])
        tsukamoto_feature_data = tsukamoto_feature_array.reshape(
            [tsukamoto_feature_array.shape[0], tsukamoto_feature_array.shape[1] * tsukamoto_feature_array.shape[2]])

        matsuo_feature_data = matsuo_feature_array.reshape(
            [tsukamoto_feature_array.shape[0], tsukamoto_feature_array.shape[1] * tsukamoto_feature_array.shape[2]])

        yamagami_feature_data = yamagami_feature_array.reshape(
            [tsukamoto_feature_array.shape[0], tsukamoto_feature_array.shape[1] * tsukamoto_feature_array.shape[2]])

        fuji_feature_data = fuji_feature_array.reshape(
            [tsukamoto_feature_array.shape[0], tsukamoto_feature_array.shape[1] * tsukamoto_feature_array.shape[2]])

        iwata_feature_data = iwata_feature_array.reshape(
            [tsukamoto_feature_array.shape[0], tsukamoto_feature_array.shape[1] * tsukamoto_feature_array.shape[2]])

    elif use_sensor_num == 1:
        tsukamoto_feature_data = copy.copy(tsukamoto_feature_array)
        matsuo_feature_data = copy.copy(matsuo_feature_array)
        yamagami_feature_data = copy.copy(yamagami_feature_array)
        fuji_feature_data = copy.copy(fuji_feature_array)
        iwata_feature_data = copy.copy(iwata_feature_array)

    # print(X_train_reshape.shape)
    tsukamoto_feature_pd = pd.DataFrame(tsukamoto_feature_data)
    matsuo_feature_pd = pd.DataFrame(matsuo_feature_data)
    yamagami_feature_pd = pd.DataFrame(yamagami_feature_data)
    fuji_feature_pd = pd.DataFrame(fuji_feature_data)
    iwata_feature_pd = pd.DataFrame(iwata_feature_data)
    # print(X_test_pd.shape)

    # 値補完
    impute(tsukamoto_feature_pd)
    impute(matsuo_feature_pd)
    impute(yamagami_feature_pd)
    impute(fuji_feature_pd)
    impute(iwata_feature_pd)

    tsukamoto_feature_data = tsukamoto_feature_pd.values
    matsuo_feature_data = matsuo_feature_pd.values
    yamagami_feature_data = yamagami_feature_pd.values
    fuji_feature_data = fuji_feature_pd.values
    iwata_feature_data = iwata_feature_pd.values

    # optunaで得たパラメータを呼び出す処理。保存したパラメータを使わないならいらない
    param_save_path = "param3_" + \
        load_data_acc_gyr + "_" + \
        str(use_sensor_num1) + "_" + str(use_sensor_num2)
    os.makedirs(param_save_path, exist_ok=True)

    #　交差検証での平均を出す際、各回の結果を保存する配列
    f1_score_array = np.empty(5)

    # dict_path = load_dic(use_sensor_num)

    #　各回の出力ラベルを保存する固定長配列。データのサイズに応じて配列サイズは変更
    arr1 = np.zeros((5, 288))

    for i in range(5):

        #　学習・テストデータのパターン別に一人抜き交差検証
        if i == 0:  # test:tsukamoto
            feature_test_data = tsukamoto_feature_data
            label_test_data = tsukamoto_label_data
            feature_train_data = np.concatenate([matsuo_feature_data,
                                                fuji_feature_data, yamagami_feature_data], 0)
            label_train_data = np.concatenate(
                [matsuo_label_data,  fuji_label_data, yamagami_label_data])

            feature_valid_data = iwata_feature_data
            label_valid_data = iwata_label_data

        elif i == 1:  # test:iwata
            feature_test_data = iwata_feature_data
            feature_train_data = np.concatenate([tsukamoto_feature_data, matsuo_feature_data,
                                                 yamagami_feature_data], 0)
            label_train_data = np.concatenate(
                [tsukamoto_label_data, matsuo_label_data, yamagami_label_data])
            label_test_data = iwata_label_data
            feature_valid_data = fuji_feature_data
            label_valid_data = fuji_label_data
        elif i == 2:  # test:fuji
            feature_test_data = fuji_feature_data
            feature_train_data = np.concatenate([iwata_feature_data, matsuo_feature_data,
                                                tsukamoto_feature_data], 0)
            label_train_data = np.concatenate(
                [iwata_label_data, matsuo_label_data,  tsukamoto_label_data])
            label_test_data = fuji_label_data
            feature_valid_data = yamagami_feature_data
            label_valid_data = yamagami_label_data
        elif i == 3:  # test:yamagami
            feature_test_data = yamagami_feature_data
            feature_train_data = np.concatenate([iwata_feature_data,
                                                fuji_feature_data, tsukamoto_feature_data], 0)
            label_train_data = np.concatenate(
                [iwata_label_data,  fuji_label_data, tsukamoto_label_data])
            label_test_data = yamagami_label_data
            feature_valid_data = matsuo_feature_data
            label_valid_data = matsuo_label_data
        elif i == 4:  # test:matsuo
            feature_test_data = matsuo_feature_data
            feature_train_data = np.concatenate([iwata_feature_data,
                                                fuji_feature_data, yamagami_feature_data], 0)
            label_train_data = np.concatenate(
                [iwata_label_data,  fuji_label_data, yamagami_label_data])
            label_test_data = matsuo_label_data
            feature_valid_data = tsukamoto_feature_data
            label_valid_data = tsukamoto_label_data
        else:
            sys.exit()

        X_train_data = feature_train_data
        y_train = label_train_data
        X_valid_data = feature_valid_data
        y_valid = label_valid_data

        trains = lgb.Dataset(X_train_data, y_train, free_raw_data=False)
        valids = lgb.Dataset(X_valid_data, y_valid, free_raw_data=False)
        # tests = lgb.Dataset(feature_test_data, label_test_data)

        # optuna使用 macrof1を最大化
        study = optuna.create_study(direction="maximize")

        study.optimize(lambda trial: bayes_objective(trial,
                                                     tsukamoto_feature_data, matsuo_feature_data, yamagami_feature_data, fuji_feature_data, iwata_feature_data,
                                                     tsukamoto_label_data, matsuo_label_data, yamagami_label_data, fuji_label_data, iwata_label_data, i), n_trials=10)

        bestparams = study.best_trial.params

        # チューニングしてないパラメータはここで指定し直す

        bestparams['objective'] = 'multiclass'
        bestparams['metric'] = 'multi_error'
        bestparams['lambda_l1'] = 0.0
        bestparams['lambda_l2'] = 0.0
        bestparams['bagging_fraction'] = 0.8
        bestparams['bagging_freq'] = 3
        bestparams['bagging_fraction'] = 0.8
        bestparams['feature_fraction'] = 0.9

        bestparams['boosting_type'] = 'gbdt'

        bestparams['learning_rate'] = 0.1
        bestparams['num_iteration'] = 100
        bestparams['num_class'] = 12
        bestparams['n_jobs'] = -1
        bestparams['early_stopping_round'] = 5
        bestparams['seed'] = 0

        save = param_save_path + "_" + str(i)
        print(save)
        np.save(save, bestparams)

        # ハイパーパラメータチューニングの各回における精度変化をプロットする部分
        fig = optuna.visualization.plot_optimization_history(study)
        fig_write_path = save + "_optimization_history.png"
        fig.write_image(fig_write_path)

        # 学習
        model = lgb.train(bestparams, trains, valid_sets=valids,
                          num_boost_round=100)

        #　テストに対する予測
        predicts = model.predict(feature_test_data)
        pred_labels = np.argmax(predicts, axis=1)
        arr1[i] = pred_labels
        #　指標はF1score
        acc = f1_score(label_test_data, pred_labels, average=average)
        np.put(f1_score_array, i, acc)

    f1_a_score = np.average(f1_score_array)
    text = "{},{}"
    write_text = text.format(use_sensor_num1, use_sensor_num2)
    acc_write = [write_text, str(f1_a_score)]
    # 出力先は適当に
    with open(write_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(acc_write)
    save = param_save_path + "_predlabel"
    np.save(save, arr1)


def main():

    # tsfreshで作った特徴量を読み出す処理。パスは適当に
    tsukamoto_label_array = np.load(
        tsukamoto_path + "/tsukamoto_feature_datalabel.npy")
    iwata_label_array = np.load(iwata_path + "/iwata_feature_datalabel.npy")
    matsuo_label_array = np.load(matsuo_path + "/matsuo_feature_datalabel.npy")
    fuji_label_array = np.load(fuji_path + "/fuji_feature_datalabel.npy")
    yamagami_label_array = np.load(
        yamagami_path + "/yamagami_feature_datalabel.npy")

    tsukamoto_feature_array = np.load(
        tsukamoto_path + "/tsukamoto_feature_datafeature.npy")
    iwata_feature_array = np.load(
        iwata_path + "/iwata_feature_datafeature.npy")
    matsuo_feature_array = np.load(
        matsuo_path + "/matsuo_feature_datafeature.npy")
    fuji_feature_array = np.load(fuji_path + "/fuji_feature_datafeature.npy")
    yamagami_feature_array = np.load(
        yamagami_path + "/yamagami_feature_datafeature.npy")

    tsukamoto_label_array = tsukamoto_label_array - 1
    yamagami_label_array = yamagami_label_array - 1
    matsuo_label_array = matsuo_label_array - 1
    fuji_label_array = fuji_label_array - 1
    iwata_label_array = iwata_label_array - 1

    # 以下学習、予測部分

    use_lightgbm_model(tsukamoto_feature_array, matsuo_feature_array, yamagami_feature_array, fuji_feature_array, iwata_feature_array,
                       tsukamoto_label_array, matsuo_label_array, yamagami_label_array, fuji_label_array, iwata_label_array, 30, 30, 100)

    for i in range(30):
        array_a = one_array_extract(tsukamoto_feature_array, i)
        array_b = one_array_extract(iwata_feature_array, i)
        array_c = one_array_extract(fuji_feature_array, i)
        array_d = one_array_extract(yamagami_feature_array, i)
        array_e = one_array_extract(matsuo_feature_array, i)
        use_lightgbm_model(array_a, array_e, array_d, array_c, array_b,
                           tsukamoto_label_array, matsuo_label_array, yamagami_label_array, fuji_label_array, iwata_label_array, 1, i, 100)

    # 重複なしコンビネーションをリスト化
    conv_list = list(itertools.combinations(range(30), 2))
    for j in range(len(conv_list)):
        array_2a = two_array_extract(
            tsukamoto_feature_array, (conv_list[j])[0], (conv_list[j])[1])
        array_2b = two_array_extract(
            iwata_feature_array, (conv_list[j])[0], (conv_list[j])[1])
        array_2c = two_array_extract(
            fuji_feature_array, (conv_list[j])[0], (conv_list[j])[1])
        array_2d = two_array_extract(
            yamagami_feature_array, (conv_list[j])[0], (conv_list[j])[1])
        array_2e = two_array_extract(
            matsuo_feature_array, (conv_list[j])[0], (conv_list[j])[1])
        use_lightgbm_model(array_2a, array_2e, array_2d, array_2c, array_2b,
                           tsukamoto_label_array, matsuo_label_array, yamagami_label_array, fuji_label_array, iwata_label_array, 2, (conv_list[j])[0], (conv_list[j])[1])


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
