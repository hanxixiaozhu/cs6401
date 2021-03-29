from sklearn import random_projection
import utils
import data_generation as dg
from sklearn.preprocessing import StandardScaler


def random_project_apply(x):
    num_features = x.shape[1]
    rps = []
    for i in range(1, num_features+1):
        rps.append(random_projection.GaussianRandomProjection(n_components=i))
    error_collections = utils.dr_apply_experiment(x, rps)
    return error_collections


def digit_data_rp_apply():
    x_train, y_train, x_test, y_test = dg.digit_data_reading()
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x = StandardScaler().fit_transform(x)
    errors = random_project_apply(x)
    return errors


def trading_data_rp_apply():
    x_train, y_train, x_test, y_test = dg.trading_data_reading()
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x = StandardScaler().fit_transform(x)
    errors = random_project_apply(x)
    return errors


if __name__ == '__main__':
    err = trading_data_rp_apply()
