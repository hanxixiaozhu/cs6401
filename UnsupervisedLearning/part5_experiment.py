from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
import data_generation as dg
import utils
import numpy as np
import config
import const


def digit_data_nn_grid():
    digit_x_train, digit_y_train, digit_x_test, digit_y_test = dg.digit_data_reading()
    nn_collections = []
    hidden_layer = []
    hidden_node = [9, 25, 64]

    for hn in hidden_node:
        tmp_hidden = []
        for i in range(1, 4):
            tmp_hidden.append(hn)
            hidden_layer.append(tuple(tmp_hidden))

    for hl in hidden_layer:
        nn = MLPClassifier(hidden_layer_sizes=hl)
        nn_collections.append(nn)

    kmeans = KMeans(n_clusters=10)
    f1_scores_collections_kmeans, roc_scores_collections_kmeans = \
        utils.cluster_nn_grid(digit_x_train, digit_y_train, config.train_data_ratio, kmeans, nn_collections)

    string_knn = f"digit data, kmeans the best hl with f1 " \
                 f"{hidden_layer[int(np.argmax(f1_scores_collections_kmeans))]} " \
                 f"with roc {hidden_layer[int(np.argmax(roc_scores_collections_kmeans))]}"

    gm = GaussianMixture(n_components=10)
    f1_scores_collections_gm, roc_scores_collections_gm = \
        utils.cluster_nn_grid(digit_x_train, digit_y_train, config.train_data_ratio, gm, nn_collections)
    string_gm = f"digit data, gaussian mixture the best hl with f1 " \
                f"{hidden_layer[int(np.argmax(f1_scores_collections_gm))]} " \
                f"with roc {hidden_layer[int(np.argmax(roc_scores_collections_gm))]}"
    print(string_knn + '\n' + string_gm)


def trading_data_nn_grid():
    trading_x_train, trading_y_train, trading_x_test, trading_y_test = dg.trading_data_reading()
    nn_collections = []
    hidden_layer = []
    hidden_node = [9, 25, 64]

    for hn in hidden_node:
        tmp_hidden = []
        for i in range(1, 4):
            tmp_hidden.append(hn)
            hidden_layer.append(tuple(tmp_hidden))

    for hl in hidden_layer:
        nn = MLPClassifier(hidden_layer_sizes=hl)
        nn_collections.append(nn)

    kmeans = KMeans(n_clusters=2)
    f1_scores_collections_kmeans, roc_scores_collections_kmeans = \
        utils.cluster_nn_grid(trading_x_train, trading_y_train, config.train_data_ratio, kmeans, nn_collections)

    string_knn = f"trading data, kmeans the best hl with f1 " \
                 f"{hidden_layer[int(np.argmax(f1_scores_collections_kmeans))]} " \
                 f"with roc {hidden_layer[int(np.argmax(roc_scores_collections_kmeans))]}"

    gm = GaussianMixture(n_components=2)
    f1_scores_collections_gm, roc_scores_collections_gm = \
        utils.cluster_nn_grid(trading_x_train, trading_y_train, config.train_data_ratio, gm, nn_collections)
    string_gm = f"trading data, gaussian mixture the best hl with f1 " \
                f"{hidden_layer[int(np.argmax(f1_scores_collections_gm))]} " \
                f"with roc {hidden_layer[int(np.argmax(roc_scores_collections_gm))]}"
    print(string_knn + '\n' + string_gm)


def cluster_grid_search():
    digit_data_nn_grid()
    trading_data_nn_grid()


def digit_data_cluster_lc_build():
    digit_x_train, digit_y_train, digit_x_test, digit_y_test = dg.digit_data_reading()

    kmeans = KMeans(n_clusters=10)
    km_nn = MLPClassifier(hidden_layer_sizes=(64, 64, 64))
    f1_km, roc_km, tr = utils.cluster_learning_curve_build(digit_x_test, digit_y_test, kmeans, km_nn)

    gm = GaussianMixture(n_components=10)
    gm_nn_f1 = MLPClassifier(hidden_layer_sizes=(64, 64))
    gm_nn_roc = MLPClassifier(hidden_layer_sizes=(64, 64, 64))
    f1_gm_f1, roc_gm_f1, tr = utils.cluster_learning_curve_build(digit_x_test, digit_y_test, gm, gm_nn_f1)
    f1_gm_roc, roc_gm_roc, tr = utils.cluster_learning_curve_build(digit_x_test, digit_y_test, gm, gm_nn_roc)

    f1_scores = [f1_km, f1_gm_f1, f1_gm_roc]
    roc_scores = [roc_km, roc_gm_f1, roc_gm_roc]

    labels = [const.km, const.gm_f1, const.gm_roc]
    utils.experiment_graph(f1_scores, const.digit_data_cluster_lc_f1_score, x_points=tr, label=labels)
    utils.experiment_graph(roc_scores, const.digit_data_cluster_lc_roc_score, x_points=tr, label=labels)


def trading_data_cluster_lc_build():
    trading_x_train, trading_y_train, trading_x_test, trading_y_test = dg.trading_data_reading()

    kmeans = KMeans(n_clusters=2)
    km_nn = MLPClassifier(hidden_layer_sizes=(25, 25, 25))
    f1_km, roc_km, tr = utils.cluster_learning_curve_build(trading_x_test, trading_y_test, kmeans, km_nn)

    gm = GaussianMixture(n_components=2)
    gm_nn_f1 = MLPClassifier(hidden_layer_sizes=(9, 9))
    f1_gm, roc_gm, tr = utils.cluster_learning_curve_build(trading_x_test, trading_y_test, gm, gm_nn_f1)

    f1_scores = [f1_km, f1_gm]
    roc_scores = [roc_km, roc_gm]

    labels = [const.km, const.gm]
    utils.experiment_graph(f1_scores, const.trading_data_cluster_lc_f1_score, x_points=tr, label=labels)
    utils.experiment_graph(roc_scores, const.trading_data_cluster_lc_roc_score, x_points=tr, label=labels)


def cluster_feature_lc_build():
    digit_data_cluster_lc_build()
    trading_data_cluster_lc_build()


if __name__ == '__main__':
    np.random.seed(config.random_seed)
    cluster_grid_search()
    # trading_data_cluster_lc_build()
    cluster_feature_lc_build()
