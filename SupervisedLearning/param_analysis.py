import const
import config
import utils
import data_generation
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


"""
in lecture
1. check boosting type
2. check svm kernel type
3. check nn

For different type learner
1. separate train and test set, use train set to do cross validation

ideas:
1. hyper-parameters might impact each other, these is a curse of dimensionality, with 3 hyperparam, test would be much
more

"""


def dt_ccp_alpha(data_x, data_y, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
                 topic=const.DecisionTreeCcpAlpha):
    """
    0.000 to 0.04 by 0.001
    """
    learners = []
    ccp_alpha_list = np.arange(0, 0.04, 0.001)
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]
    for ccp_alpha in ccp_alpha_list:
        ind_lrn = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        learners.append(ind_lrn)
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, ccp_alpha_list, evaluation_methods, topic=topic)


def dt_max_depth(data_x, data_y, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
                 topic=const.DecisionTreeMaxDepth):
    """
    test decision tree with different max depth from 1 to 11
    """
    learners = []
    depth_list = np.arange(1, 31, 2)
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]
    for depth in depth_list:
        ind_lrn = DecisionTreeClassifier(max_depth=depth)
        learners.append(ind_lrn)
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, depth_list, evaluation_methods, topic=topic)


def adaboost_base(data_x, data_y, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
                  topic=const.adaBoostMaxDepth):
    """
    test with different base tree learner, with n_estimators as 20
    adaboost should not be able to overfit
    """
    learners = []
    depth_list = np.arange(1, 31, 2)
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]
    for depth in depth_list:
        base_learner = DecisionTreeClassifier(max_depth=depth)
        ind_lrn = AdaBoostClassifier(base_learner, n_estimators=20)
        learners.append(ind_lrn)
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, depth_list, evaluation_methods, topic=topic)


def adaboost_num_estimator(data_x, data_y, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
                           topic=const.adaBoostNumEstimator):
    """
    test with different number of estimators, with max depth of estimator of 9

    """
    learners = []
    num_estimator_list = np.arange(5, 105, 5)
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]
    for num_estimator in num_estimator_list:
        base_learner = DecisionTreeClassifier(max_depth=9)
        ind_lrn = AdaBoostClassifier(base_learner, n_estimators=num_estimator)
        learners.append(ind_lrn)
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, num_estimator_list, evaluation_methods, topic=topic)


def mlp_hid_layer_node(data_x, data_y, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
                       topic=const.mlp_num_node):
    """
    test for 1 hidden layer with n nodes
    """
    learners = []
    num_nodes_list = np.arange(5, 35, 5)
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]
    for hid_size in num_nodes_list:
        ind_lrn = MLPClassifier(hidden_layer_sizes=hid_size)
        learners.append(ind_lrn)
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, num_nodes_list, evaluation_methods, topic=topic)


def mlp_hid_layer_num(data_x, data_y, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
                      topic=const.mlp_num_layer):
    """
    test for n hidden layer with 5 nodes
    """
    learners = []
    num_layer_list = np.arange(1, 10)
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]

    for hid_layer in num_layer_list:
        ind_lrn = MLPClassifier(hidden_layer_sizes=tuple([9] * hid_layer))
        learners.append(ind_lrn)
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, num_layer_list, evaluation_methods, topic=topic)


def knn_num_neighbor(data_x, data_y, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
                     topic=const.knn_num_neighbor):
    """
    test for n neighbors
    """
    learners = []
    num_neighbor_list = np.arange(1, 100, 3)
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]

    for num_neighbor in num_neighbor_list:
        ind_base = KNeighborsClassifier(n_neighbors=num_neighbor)
        learners.append(ind_base)
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, num_neighbor_list, evaluation_methods, topic=topic)


def svm_kernel(data_x, data_y, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
               topic=const.svm_kernel):
    """
    test for different kernels
    """

    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]
    svm_linear = SVC(kernel='linear')
    svm_poly = SVC(kernel='poly')  # degree==3 gamma=='sacle'
    svm_rbf = SVC(kernel='rbf')  # gamma=='sacle'
    svm_sigmoid = SVC(kernel='sigmoid')  # gamma=='sacle'
    learners = [svm_linear, svm_poly, svm_rbf, svm_sigmoid]
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, np.arange(4), evaluation_methods, topic=topic)


def svm_c(data_x, data_y, kernel_type, num_fold=3, split_seed=None, evaluation_methods=const.f1, shuffle=True,
          topic=const.svm_c):
    """
    test for different kernels
    """

    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]
    c_list = np.arange(1, 15, 1)

    learners = []
    for c in c_list:
        ind_lrn = SVC(C=c, kernel=kernel_type)
        learners.append(ind_lrn)
    train_score, test_score = utils.compare_learners(data_x, data_y, learners, num_fold=num_fold,
                                                     split_seed=split_seed, evaluation_methods=evaluation_methods,
                                                     shuffle=shuffle)
    utils.learner_compare_graphing(train_score, test_score, c_list, evaluation_methods, topic=topic)


def digit_data_experiment():
    data_x, data_y = data_generation.digit_x_train, data_generation.digit_y_train
    dt_ccp_alpha(data_x, data_y, evaluation_methods=const.f1)
    dt_max_depth(data_x, data_y)
    adaboost_base(data_x, data_y)
    adaboost_num_estimator(data_x, data_y)
    mlp_hid_layer_node(data_x, data_y)
    mlp_hid_layer_num(data_x, data_y)
    knn_num_neighbor(data_x, data_y)
    svm_kernel(data_x, data_y)
    svm_c(data_x, data_y, 'poly')


def cancer_data_experiment():
    data_x, data_y = data_generation.cancer_x_train, data_generation.cancer_y_train
    dt_ccp_alpha(data_x, data_y, evaluation_methods=const.f1_micro)
    dt_max_depth(data_x, data_y, evaluation_methods=const.f1_micro)
    adaboost_base(data_x, data_y, evaluation_methods=const.f1_micro)
    adaboost_num_estimator(data_x, data_y, evaluation_methods=const.f1_micro)
    mlp_hid_layer_node(data_x, data_y, evaluation_methods=const.f1_micro)
    mlp_hid_layer_num(data_x, data_y, evaluation_methods=const.f1_micro)
    knn_num_neighbor(data_x, data_y, evaluation_methods=const.f1_micro)
    svm_kernel(data_x, data_y, evaluation_methods=const.f1_micro)
    svm_c(data_x, data_y, 'linear', evaluation_methods=const.f1_micro)


if __name__ == '__main__':
    np.random.seed(config.random_seed)
    digit_data_experiment()
    cancer_data_experiment()
