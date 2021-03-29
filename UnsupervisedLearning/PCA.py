from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils
import data_generation as dg


def pca_apply(x):
    num_features = x.shape[1]
    pcas = []
    for i in range(1, num_features+1):
        pcas.append(PCA(n_components=i))
    error_collections = utils.dr_apply_experiment(x, pcas)
    return error_collections


def digit_data_pca_apply():
    x_train, y_train, x_test, y_test = dg.digit_data_reading()
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x = StandardScaler().fit_transform(x)
    errors = pca_apply(x)
    return errors


def trading_data_pca_apply():
    x_train, y_train, x_test, y_test = dg.trading_data_reading()
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x = StandardScaler().fit_transform(x)
    errors = pca_apply(x)
    return errors


if __name__ == '__main__':
    err = digit_data_pca_apply()
