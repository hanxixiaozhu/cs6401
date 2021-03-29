from sklearn import datasets
import numpy as np
import config
import pandas as pd
import const


def split_data(data_x, data_y, train_ratio):
    # data_x, data_y = datasets.load_digits(return_X_y=True, as_frame=True)
    data_length = data_x.shape[0]
    data_idx = np.arange(data_length)
    np.random.shuffle(data_idx)
    data_train_length = int(data_length * train_ratio)
    train_idx = data_idx[: data_train_length]
    test_idx = data_idx[data_train_length:]

    data_x_train = data_x.iloc[train_idx]
    data_y_train = data_y.iloc[train_idx]

    data_x_test = data_x.iloc[test_idx]
    data_y_test = data_y.iloc[test_idx]
    return data_x_train.reset_index(drop=True), \
        data_y_train.reset_index(drop=True), data_x_test.reset_index(drop=True), data_y_test.reset_index(drop=True)


def digit_data_reading():
    data1_x, data1_y = datasets.load_digits(return_X_y=True, as_frame=True)
    digit_x_train, digit_y_train, digit_x_test, digit_y_test = split_data(data1_x, data1_y, config.train_data_ratio)
    return digit_x_train, digit_y_train, digit_x_test, digit_y_test


def trading_data_reading(file_path='data/trading.csv'):
    raw_data = pd.read_csv(file_path)
    raw_data_y = raw_data[const.is_profit]
    raw_data_x = raw_data[const.x_columns]
    trading_x_train, trading_y_train, trading_x_test, trading_y_test = split_data(raw_data_x, raw_data_y,
                                                                                  config.train_data_ratio)
    return trading_x_train, trading_y_train, trading_x_test, trading_y_test
