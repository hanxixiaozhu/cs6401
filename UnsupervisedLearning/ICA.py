from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import utils
import numpy as np
from scipy.stats import kurtosis
import data_generation as dg


def ica_apply(x):
    num_features = x.shape[1]
    icas = []
    for i in range(1, num_features+1):
        icas.append(FastICA(n_components=i))
    error_collections = utils.dr_apply_experiment(x, icas)
    return error_collections


def digit_data_ica_apply():
    x_train, y_train, x_test, y_test = dg.digit_data_reading()
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x = StandardScaler().fit_transform(x)
    errors = ica_apply(x)
    return errors


def trading_data_ica_apply():
    x_train, y_train, x_test, y_test = dg.trading_data_reading()
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x = StandardScaler().fit_transform(x)
    errors = ica_apply(x)
    return errors


def avg_kurtosis_ica(x):
    num_features = x.shape[1]
    avg_kurtosis_collection = []
    for i in range(1, num_features+1):
        ica = FastICA(n_components=i)
        transformed_x = ica.fit_transform(x)
        avg_kurtosis = np.average(kurtosis(transformed_x))
        avg_kurtosis_collection.append(avg_kurtosis)
    return avg_kurtosis_collection


if __name__ == '__main__':
    err = digit_data_ica_apply()
