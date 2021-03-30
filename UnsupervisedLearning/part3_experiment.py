from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
import utils
import l1_regularization
import kmeans
import gaussian_mixture
import data_generation as dg
import pandas as pd
import const
import numpy as np
import config


def pca_apply(x_train, x_test, n_component):
    x = utils.x_preprocess(x_train, x_test)
    x_reduce = PCA(n_components=n_component).fit_transform(x)
    return x_reduce


def ica_apply(x_train, x_test, n_component):
    x = utils.x_preprocess(x_train, x_test)
    x_reduce = FastICA(n_components=n_component).fit_transform(x)
    return x_reduce


def rp_apply(x_train, x_test, n_component):
    x = utils.x_preprocess(x_train, x_test)
    x_reduce = random_projection.GaussianRandomProjection(n_components=n_component).fit_transform(x)
    return x_reduce


def l1_apply(x_train, y_train, x_test, y_test):
    x, y = utils.train_test_combine(x_train, y_train, x_test, y_test)
    x_selected, selected_x_indice = l1_regularization.l1_logistic_selection(x, y)
    return x_selected


def kmeans_part3_experiment(x_train, y_train, x_test, y_test, cluster_nums, return_accuracy_score, graph, title):
    si_score_collection, mi_collection, accuracy_collection = kmeans.knn_experiment(x_train, y_train, x_test, y_test,
                                                                                    cluster_nums,
                                                                                    return_accuracy_score, graph, title)
    return si_score_collection, mi_collection, accuracy_collection


def en_part3_experiment(x_train, y_train, x_test, y_test, cluster_nums, return_accuracy_score, graph, title):
    si_score_collection, mi_collection, accuracy_collection = gaussian_mixture.em_experiment(x_train, y_train, x_test,
                                                                                             y_test, cluster_nums,
                                                                                             return_accuracy_score,
                                                                                             graph, title)
    return si_score_collection, mi_collection, accuracy_collection


def digit_knn():
    x_train, y_train, x_test, y_test = dg.digit_data_reading()
    y_combine = pd.concat([y_train, y_test])

    x_pca = pca_apply(x_train, x_test, 50)
    x_ica = ica_apply(x_train, x_test, 59)
    x_rp = rp_apply(x_train, x_test, 50)
    x_l1 = l1_apply(x_train, y_train, x_test, y_test)

    x_test_fake = pd.DataFrame()
    y_test_fake = pd.Series()

    cluster_nums = range(2, 20, 2)
    kmeans_part3_experiment(x_pca, y_combine, x_test_fake, y_test_fake, cluster_nums, False, True,
                            const.PCA+'_'+const.digit_data_knn_silhouette_mi_score)
    kmeans_part3_experiment(x_ica, y_combine, x_test_fake, y_test_fake, cluster_nums, False, True,
                            const.ICA + '_' + const.digit_data_knn_silhouette_mi_score)
    kmeans_part3_experiment(x_rp, y_combine, x_test_fake, y_test_fake, cluster_nums, False, True,
                            const.RandomProject+'_'+const.digit_data_knn_silhouette_mi_score)
    kmeans_part3_experiment(x_l1, y_combine, x_test_fake, y_test_fake, cluster_nums, False, True,
                            const.L1+'_'+const.digit_data_knn_silhouette_mi_score)


def trading_knn():
    x_train, y_train, x_test, y_test = dg.trading_data_reading()
    y_combine = pd.concat([y_train, y_test])

    x_pca = pca_apply(x_train, x_test, 17)
    x_ica = ica_apply(x_train, x_test, 19)
    x_rp = rp_apply(x_train, x_test, 17)
    x_l1 = l1_apply(x_train, y_train, x_test, y_test)

    x_test_fake = pd.DataFrame()
    y_test_fake = pd.Series()

    cluster_nums = range(2, 5, 1)
    kmeans_part3_experiment(x_pca, y_combine, x_test_fake, y_test_fake, cluster_nums, True, True,
                            const.PCA+'_'+const.trading_data_knn_silhouette_mi_score)
    kmeans_part3_experiment(x_ica, y_combine, x_test_fake, y_test_fake, cluster_nums, True, True,
                            const.ICA + '_' + const.trading_data_knn_silhouette_mi_score)
    kmeans_part3_experiment(x_rp, y_combine, x_test_fake, y_test_fake, cluster_nums, True, True,
                            const.RandomProject+'_'+const.trading_data_knn_silhouette_mi_score)
    kmeans_part3_experiment(x_l1, y_combine, x_test_fake, y_test_fake, cluster_nums, True, True,
                            const.L1+'_'+const.trading_data_knn_silhouette_mi_score)


def digit_em():
    x_train, y_train, x_test, y_test = dg.digit_data_reading()
    y_combine = pd.concat([y_train, y_test])

    x_pca = pca_apply(x_train, x_test, 50)
    x_ica = ica_apply(x_train, x_test, 59)
    x_rp = rp_apply(x_train, x_test, 50)
    x_l1 = l1_apply(x_train, y_train, x_test, y_test)

    x_test_fake = pd.DataFrame()
    y_test_fake = pd.Series()

    cluster_nums = range(2, 20, 2)
    en_part3_experiment(x_pca, y_combine, x_test_fake, y_test_fake, cluster_nums, False, True,
                        const.PCA+'_'+const.digit_data_em_silhouette_mi_score)
    en_part3_experiment(x_ica, y_combine, x_test_fake, y_test_fake, cluster_nums, False, True,
                        const.ICA + '_' + const.digit_data_em_silhouette_mi_score)
    en_part3_experiment(x_rp, y_combine, x_test_fake, y_test_fake, cluster_nums, False, True,
                        const.RandomProject+'_'+const.digit_data_em_silhouette_mi_score)
    en_part3_experiment(x_l1, y_combine, x_test_fake, y_test_fake, cluster_nums, False, True,
                        const.L1+'_'+const.digit_data_em_silhouette_mi_score)


def trading_em():
    x_train, y_train, x_test, y_test = dg.trading_data_reading()
    y_combine = pd.concat([y_train, y_test])

    x_pca = pca_apply(x_train, x_test, 17)
    x_ica = ica_apply(x_train, x_test, 19)
    x_rp = rp_apply(x_train, x_test, 17)
    x_l1 = l1_apply(x_train, y_train, x_test, y_test)

    x_test_fake = pd.DataFrame()
    y_test_fake = pd.Series()

    cluster_nums = range(2, 5, 1)
    en_part3_experiment(x_pca, y_combine, x_test_fake, y_test_fake, cluster_nums, True, True,
                        const.PCA+'_'+const.trading_data_em_silhouette_mi_score)
    en_part3_experiment(x_ica, y_combine, x_test_fake, y_test_fake, cluster_nums, True, True,
                        const.ICA + '_' + const.trading_data_em_silhouette_mi_score)
    en_part3_experiment(x_rp, y_combine, x_test_fake, y_test_fake, cluster_nums, True, True,
                        const.RandomProject+'_'+const.trading_data_em_silhouette_mi_score)
    en_part3_experiment(x_l1, y_combine, x_test_fake, y_test_fake, cluster_nums, True, True,
                        const.L1+'_'+const.trading_data_em_silhouette_mi_score)


if __name__ == '__main__':
    np.random.seed(config.random_seed)
    print("start")
    digit_knn()
    print("knn")
    digit_em()
    print("em")
    trading_knn()
    print("knn")
    trading_em()
    print("em")
