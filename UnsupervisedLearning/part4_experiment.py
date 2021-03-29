from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import utils
import data_generation as dg
from sklearn.neural_network import MLPClassifier
import config
import numpy as np
import l1_regularization
import const


def digit_dr_nn_grid(dr):
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

    f1_scores_collections, roc_scores_collections = \
        utils.grid_compare(digit_x_train, digit_y_train, 1-config.train_data_ratio, dr, nn_collections)
    return f1_scores_collections, roc_scores_collections, hidden_layer


def trading_dr_nn_grid(dr):
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

    f1_scores_collections, roc_scores_collections = \
        utils.grid_compare(trading_x_train, trading_y_train, 1 - config.train_data_ratio, dr, nn_collections)
    return f1_scores_collections, roc_scores_collections, hidden_layer


def digit_grid_search():
    pca = PCA(n_components=50)
    pca_grid_f1, pca_grid_roc, nn_hl = digit_dr_nn_grid(pca)

    ica = FastICA(n_components=59)
    ica_grid_f1, ica_grid_roc, nn_hl = digit_dr_nn_grid(ica)

    rp = GaussianRandomProjection(n_components=50)
    rp_grid_f1, rp_grid_roc, nn_hl = digit_dr_nn_grid(rp)
    string = f"digit pca best f1 nn hl {nn_hl[int(np.argmax(pca_grid_f1))]}, " \
             f"roc nn hl {nn_hl[int(np.argmax(pca_grid_f1))]} \n " \
             f"digit ica best f1 nn hl {nn_hl[int(np.argmax(ica_grid_f1))]}, " \
             f"roc nn hl {nn_hl[int(np.argmax(ica_grid_roc))]} \n" \
             f"digit rca best f1 nn hl {nn_hl[int(np.argmax(rp_grid_f1))]}, " \
             f"roc nn hl {nn_hl[int(np.argmax(rp_grid_roc))]} \n"
    print(string)


def digit_l1_grid_search():
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
    f1_scores_collections, roc_scores_collections, selected_x_indice = \
        utils.l1_nn_grid(digit_x_train, digit_y_train, config.train_data_ratio, nn_collections)
    string = f"digit l1 best f1 nn hl {hidden_layer[int(np.argmax(f1_scores_collections))]}, " \
             f"roc nn hl {hidden_layer[int(np.argmax(roc_scores_collections))]}"
    print(string)


def trading_grid_search():
    pca = PCA(n_components=17)
    pca_grid_f1, pca_grid_roc, nn_hl = trading_dr_nn_grid(pca)

    ica = FastICA(n_components=19)
    ica_grid_f1, ica_grid_roc, nn_hl = trading_dr_nn_grid(ica)

    rp = GaussianRandomProjection(n_components=17)
    rp_grid_f1, rp_grid_roc, nn_hl = trading_dr_nn_grid(rp)

    string = f"trading pca best f1 nn hl {nn_hl[int(np.argmax(pca_grid_f1))]}, " \
             f"roc nn hl {nn_hl[int(np.argmax(pca_grid_roc))]} \n" \
             f"trading ica best f1 nn hl {nn_hl[int(np.argmax(ica_grid_f1))]}, " \
             f"roc nn hl {nn_hl[int(np.argmax(ica_grid_roc))]} \n" \
             f"trading rca best f1 nn hl {nn_hl[int(np.argmax(rp_grid_f1))]}, " \
             f"roc nn hl {nn_hl[int(np.argmax(rp_grid_roc))]} \n"
    print(string)


def trading_l1_grid_search():
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
    f1_scores_collections, roc_scores_collections, selected_x_indice = \
        utils.l1_nn_grid(trading_x_train, trading_y_train, config.train_data_ratio, nn_collections)
    string = f"trading l1 best f1 nn hl {hidden_layer[int(np.argmax(f1_scores_collections))]}, " \
             f"roc nn hl {hidden_layer[int(np.argmax(roc_scores_collections))]}"
    print(string)


def grid_search():
    digit_grid_search()
    trading_grid_search()
    digit_l1_grid_search()
    trading_l1_grid_search()


def digit_lc_build():
    digit_x_train, digit_y_train, digit_x_test, digit_y_test = dg.digit_data_reading()

    pca = PCA(n_components=50)
    nn_pca_f1 = MLPClassifier(hidden_layer_sizes=64)
    nn_pca_roc = MLPClassifier(hidden_layer_sizes=(25, 25))
    f1_pca_f1, roc_pca_f1, tr = utils.build_learning_curve(digit_x_test, digit_y_test, pca, nn_pca_f1)
    f1_pca_roc, roc_pca_roc, tr = utils.build_learning_curve(digit_x_test, digit_y_test, pca, nn_pca_roc)

    ica = FastICA(n_components=59)
    nn_ica_f1 = MLPClassifier(hidden_layer_sizes=(64, 64))
    nn_ica_roc = MLPClassifier(hidden_layer_sizes=64)
    f1_ica_f1, roc_ica_f1, tr = utils.build_learning_curve(digit_x_test, digit_y_test, ica, nn_ica_f1)
    f1_ica_roc, roc_ica_roc, tr = utils.build_learning_curve(digit_x_test, digit_y_test, ica, nn_ica_roc)

    rca = GaussianRandomProjection(n_components=50)
    nn_rca = MLPClassifier(hidden_layer_sizes=(64, 64, 64))
    f1_rca, roc_rca, tr = utils.build_learning_curve(digit_x_test, digit_y_test, rca, nn_rca)

    x_selected, x_selected_indice = l1_regularization.l1_logistic_selection(digit_x_train, digit_y_train)
    digit_x_test_selected = digit_x_test.iloc[:, x_selected_indice]
    nn_l1_f1 = MLPClassifier(hidden_layer_sizes=(64, 64))
    nn_l1_roc = MLPClassifier(hidden_layer_sizes=(64, 64, 64))
    f1_l1_f1, roc_l1_f1, tr = utils.build_learning_curve_l1(digit_x_test_selected, digit_y_test, nn_l1_f1)
    f1_l1_roc, roc_l1_roc, tr = utils.build_learning_curve_l1(digit_x_test_selected, digit_y_test, nn_l1_roc)

    f1_scores = [f1_pca_f1, f1_pca_roc, f1_ica_f1, f1_ica_roc, f1_rca, f1_l1_f1, f1_l1_roc]
    roc_scores = [roc_pca_f1, roc_pca_roc, roc_ica_f1, roc_ica_roc, roc_rca, roc_l1_f1, roc_l1_roc]

    labels = [const.pca_f1, const.pca_roc, const.ica_f1, const.ica_roc, const.rca, const.l1_f1, const.l1_roc]
    utils.experiment_graph(f1_scores, const.digit_data_lc_f1_score, x_points=tr, label=labels)
    utils.experiment_graph(roc_scores, const.digit_data_lc_roc_score, x_points=tr, label=labels)
    # return f1_pca_f1, roc_pca_f1


def trading_lc_build():
    trading_x_train, trading_y_train, trading_x_test, trading_y_test = dg.trading_data_reading()

    pca = PCA(n_components=17)
    nn_pca_f1 = MLPClassifier(hidden_layer_sizes=9)
    nn_pca_roc = MLPClassifier(hidden_layer_sizes=25)
    f1_pca_f1, roc_pca_f1, tr = utils.build_learning_curve(trading_x_test, trading_y_test, pca, nn_pca_f1)
    f1_pca_roc, roc_pca_roc, tr = utils.build_learning_curve(trading_x_test, trading_y_test, pca, nn_pca_roc)

    ica = FastICA(n_components=19)
    nn_ica = MLPClassifier(hidden_layer_sizes=(64, 64, 64))
    f1_ica, roc_ica, tr = utils.build_learning_curve(trading_x_test, trading_y_test, ica, nn_ica)

    rca = GaussianRandomProjection(n_components=17)
    nn_rca = MLPClassifier(hidden_layer_sizes=64)
    f1_rca, roc_rca, tr = utils.build_learning_curve(trading_x_test, trading_y_test, rca, nn_rca)

    x_selected, x_selected_indice = l1_regularization.l1_logistic_selection(trading_x_train, trading_y_train)
    digit_x_test_selected = trading_x_test.iloc[:, x_selected_indice]
    nn_l1_f1 = MLPClassifier(hidden_layer_sizes=64)
    nn_l1_roc = MLPClassifier(hidden_layer_sizes=(9, 9, 9))
    f1_l1_f1, roc_l1_f1, tr = utils.build_learning_curve_l1(digit_x_test_selected, trading_y_test, nn_l1_f1)
    f1_l1_roc, roc_l1_roc, tr = utils.build_learning_curve_l1(digit_x_test_selected, trading_y_test, nn_l1_roc)

    f1_scores = [f1_pca_f1, f1_pca_roc, f1_ica, f1_rca, f1_l1_f1, f1_l1_roc]
    roc_scores = [roc_pca_f1, roc_pca_roc, roc_ica, roc_rca, roc_l1_f1, roc_l1_roc]

    labels = [const.pca_f1, const.pca_roc, const.ica, const.rca, const.l1_f1, const.l1_roc]
    utils.experiment_graph(f1_scores, const.trading_data_lc_f1_score, x_points=tr, label=labels)
    utils.experiment_graph(roc_scores, const.trading_data_lc_roc_score, x_points=tr, label=labels)


def lc_curve_build():
    digit_lc_build()
    trading_lc_build()


if __name__ == '__main__':
    np.random.seed(config.random_seed)
    grid_search()
    lc_curve_build()
