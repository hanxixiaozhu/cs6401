from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils
import data_generation as dg
import const
import matplotlib.pyplot as plt


def pca_apply(x):
    num_features = x.shape[1]
    pcas = []
    for i in range(1, num_features+1):
        pcas.append(PCA(n_components=i))
    error_collections = utils.dr_apply_experiment(x, pcas)
    return error_collections


def pca_eigen_value(x, title1, title2):
    num_features = x.shape[1]
    pcas = [PCA(n_components=num_features)]
    eigens = utils.dr_apply_experiment(x, pcas, return_type=const.eigenvalues)[0]
    eigens_ratio = utils.dr_apply_experiment(x, pcas, return_type=const.eigenvalues_ratio)[0]
    eigen_idx = list(range(1, num_features+1))
    plt.plot(eigen_idx, eigens)
    plt.title(title1)
    plt.savefig(title1)
    plt.close()

    plt.plot(eigen_idx, eigens_ratio)
    plt.title(title2)
    plt.savefig(title2)
    plt.close()


def digit_data_pca_apply():
    x_train, y_train, x_test, y_test = dg.digit_data_reading()
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x = StandardScaler().fit_transform(x)
    # errors = pca_apply(x)
    pca_eigen_value(x, "Digit Data PCA Eigenvalue dist", "Digit Data PCA Eigenvalue Ratio dist")
    # return errors


def trading_data_pca_apply():
    x_train, y_train, x_test, y_test = dg.trading_data_reading()
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x = StandardScaler().fit_transform(x)
    # errors = pca_apply(x)
    pca_eigen_value(x, "Trading Data PCA Eigenvalue dist", "Trading Data PCA Eigenvalue Ratio dist")
    # return errors


if __name__ == '__main__':
    digit_data_pca_apply()
    trading_data_pca_apply()
