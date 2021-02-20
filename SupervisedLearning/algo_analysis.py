"""
From param_analysis, we choose a set of parameters that seems to be the best fit for the dataset
"""
import const
import utils
import data_generation
from sklearn.tree import DecisionTreeClassifier
import config
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def decision_tree(data_x, data_y, learner, test_ratio: np.array, evaluation_methods=const.f1, shuffle=True,
                  topic=const.learning_curve+'_'+const.DecisionTree):
    """

    """
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]

    if len(test_ratio) == 0 or test_ratio is None:
        test_ratio = config.default_test_ratio
    # learner = DecisionTreeClassifier(ccp_alpha=0.002, max_depth=18)
    train_score, test_score = utils.learning_curve(data_x, data_y, learner, test_ratio, evaluation_methods, shuffle)
    consolidate_train = utils.build_learning_score(train_score, evaluation_methods)
    consolidate_test = utils.build_learning_score(test_score, evaluation_methods)
    utils.learning_curve_graphing(consolidate_train, consolidate_test, test_ratio, evaluation_methods, topic)


def adaboost(data_x, data_y, learner, test_ratio: np.array, evaluation_methods=const.f1, shuffle=True,
             topic=const.learning_curve+'_'+const.adaboost):
    """

    """
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]

    if len(test_ratio) == 0 or test_ratio is None:
        test_ratio = config.default_test_ratio
    # base_tree = DecisionTreeClassifier(max_depth=9)
    # learner = AdaBoostClassifier(base_tree, n_estimators=90)
    train_score, test_score = utils.learning_curve(data_x, data_y, learner, test_ratio, evaluation_methods, shuffle)
    consolidate_train = utils.build_learning_score(train_score, evaluation_methods)
    consolidate_test = utils.build_learning_score(test_score, evaluation_methods)
    utils.learning_curve_graphing(consolidate_train, consolidate_test, test_ratio, evaluation_methods, topic)


def svm(data_x, data_y, learner, test_ratio: np.array, evaluation_methods=const.f1, shuffle=True,
        topic=const.learning_curve+'_'+const.svm):
    """

    """
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]

    if len(test_ratio) == 0 or test_ratio is None:
        test_ratio = config.default_test_ratio
    # learner = SVC(kernel='poly')
    train_score, test_score = utils.learning_curve(data_x, data_y, learner, test_ratio, evaluation_methods, shuffle)
    consolidate_train = utils.build_learning_score(train_score, evaluation_methods)
    consolidate_test = utils.build_learning_score(test_score, evaluation_methods)
    utils.learning_curve_graphing(consolidate_train, consolidate_test, test_ratio, evaluation_methods, topic)


def mlp(data_x, data_y, learner, test_ratio: np.array, evaluation_methods=const.f1, shuffle=True,
        topic=const.learning_curve+'_'+const.mlp):
    """

    """
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]

    if len(test_ratio) == 0 or test_ratio is None:
        test_ratio = config.default_test_ratio
    # learner = MLPClassifier(hidden_layer_sizes=(20, 20, 20))
    train_score, test_score = utils.learning_curve(data_x, data_y, learner, test_ratio, evaluation_methods, shuffle)
    consolidate_train = utils.build_learning_score(train_score, evaluation_methods)
    consolidate_test = utils.build_learning_score(test_score, evaluation_methods)
    utils.learning_curve_graphing(consolidate_train, consolidate_test, test_ratio, evaluation_methods, topic)


def knn(data_x, data_y, learner, test_ratio: np.array, evaluation_methods=const.f1, shuffle=True,
        topic=const.learning_curve+'_'+const.knn):
    """

    """
    if not isinstance(evaluation_methods, list):
        evaluation_methods = [evaluation_methods]

    if len(test_ratio) == 0 or test_ratio is None:
        test_ratio = config.default_test_ratio
    # learner = KNeighborsClassifier(n_neighbors=2)
    train_score, test_score = utils.learning_curve(data_x, data_y, learner, test_ratio, evaluation_methods, shuffle)
    consolidate_train = utils.build_learning_score(train_score, evaluation_methods)
    consolidate_test = utils.build_learning_score(test_score, evaluation_methods)
    utils.learning_curve_graphing(consolidate_train, consolidate_test, test_ratio, evaluation_methods, topic)


def digit_data_experiment():
    data2_x, data2_y = data_generation.digit_x_test, data_generation.digit_y_test
    exp_test_ratio = np.arange(0.05, 1, 0.05)
    tree = DecisionTreeClassifier(ccp_alpha=0.003, max_depth=16)
    decision_tree(data2_x, data2_y, tree, exp_test_ratio)
    base_tree = DecisionTreeClassifier(max_depth=9)
    adatree = AdaBoostClassifier(base_tree, n_estimators=65)
    adaboost(data2_x, data2_y, adatree, exp_test_ratio)
    svm_learner = SVC(kernel='poly')
    svm(data2_x, data2_y, svm_learner, exp_test_ratio)
    mlp_learner = MLPClassifier(hidden_layer_sizes=(25, 25))
    mlp(data2_x, data2_y, mlp_learner, exp_test_ratio)
    knn_learner = KNeighborsClassifier(n_neighbors=1)
    knn(data2_x, data2_y, knn_learner, exp_test_ratio)


def cancer_data_experiment():
    data_x, data_y = data_generation.cancer_x_test, data_generation.cancer_y_test
    exp_test_ratio = np.arange(0.05, 1, 0.05)
    tree = DecisionTreeClassifier(ccp_alpha=0.009, max_depth=23)
    decision_tree(data_x, data_y, tree, exp_test_ratio, evaluation_methods=const.f1_micro)
    base_tree = DecisionTreeClassifier(max_depth=2)
    adatree = AdaBoostClassifier(base_tree, n_estimators=5)
    adaboost(data_x, data_y, adatree, exp_test_ratio, evaluation_methods=const.f1_micro)
    svm_learner = SVC(C=5, kernel='linear')
    svm(data_x, data_y, svm_learner, exp_test_ratio, evaluation_methods=const.f1_micro)
    mlp_learner = MLPClassifier(hidden_layer_sizes=tuple([15]*6))
    mlp(data_x, data_y, mlp_learner, exp_test_ratio, evaluation_methods=const.f1_micro)
    knn_learner = KNeighborsClassifier(n_neighbors=7)
    knn(data_x, data_y, knn_learner, exp_test_ratio, evaluation_methods=const.f1_micro)
    data_full_x, data_full_y = data_generation.data3_x, data_generation.data3_y
    mlp(data_full_x, data_full_y, mlp_learner, exp_test_ratio, evaluation_methods=const.f1_micro,
        topic=const.learning_curve+'_'+const.mlp+'full_data')


if __name__ == '__main__':
    np.random.seed(config.random_seed)
    digit_data_experiment()
    cancer_data_experiment()
