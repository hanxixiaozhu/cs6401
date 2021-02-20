from sklearn import datasets
import numpy as np
import config


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
    return data_x_train.reset_index(drop=True), data_y_train.reset_index(drop=True), \
           data_x_test.reset_index(drop=True), data_y_test.reset_index(drop=True)


np.random.seed(config.random_seed)
# data1_x, data1_y = datasets.load_iris(return_X_y=True, as_frame=True)
data2_x, data2_y = datasets.load_digits(return_X_y=True, as_frame=True)
data3_x, data3_y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

digit_x_train, digit_y_train, digit_x_test, digit_y_test = split_data(data2_x, data2_y, config.train_data_ratio)
cancer_x_train, cancer_y_train, cancer_x_test, cancer_y_test = split_data(data3_x, data3_y, config.train_data_ratio)
