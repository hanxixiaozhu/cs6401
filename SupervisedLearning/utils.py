import const
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import collections
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# import time


def evaluation(y_predict, y_true, method):
    """
    return evaluation result
    """
    if method == const.f1:
        # macro, unweighted avg of each class f1
        return f1_score(y_true, y_predict, average='macro')
    elif method == const.f1_micro:
        # micro, weighted avg of each class f1
        return f1_score(y_true, y_predict, average='micro')
    elif method == const.accuracy:
        return accuracy_score(y_true, y_predict)
    else:
        raise ValueError(f"{method} not supported yet")


def single_set_evaluation(train_x, test_x, train_y, test_y, learners, evaluation_methods):
    """
    return 2 dicts, training and testing , key is method, and value is a list, the list contains each learner result.
    """
    for incident in learners:
        incident.fit(train_x, train_y)
    train_y = train_y.values
    test_y = test_y.values
    learners_test_outputs = []
    learners_train_outputs = []
    for incident in learners:
        test_output = incident.predict(test_x)
        train_output = incident.predict(train_x)
        learners_test_outputs.append(test_output)
        learners_train_outputs.append(train_output)

    test_scores = collections.defaultdict(list)
    train_scores = collections.defaultdict(list)
    for method in evaluation_methods:
        for output in learners_test_outputs:
            test_score = evaluation(output, test_y, method)
            test_scores[method].append(test_score)
        for output in learners_train_outputs:
            train_score = evaluation(output, train_y, method)
            train_scores[method].append(train_score)
    return train_scores, test_scores


def compare_learners(data_x, data_y, learners, num_fold=3, split_seed=None, evaluation_methods=const.f1,
                     shuffle=True):
    """
    return 2 lists, train list and test list, each element is a fold (evaluation) result (each element is a dict)
    learners: list of sklearn learners, it can be the same type learner but with hyperparams
    evaluation_methods: list of learning method
    """

    if not isinstance(learners, list):
        learners = [learners]
    # if not isinstance(evaluation_methods, list):
    #     evaluation_methods = [evaluation_methods]
    kf = KFold(n_splits=num_fold, shuffle=shuffle, random_state=split_seed)
    test_fold_score = []
    train_fold_score = []
    for train_idx, test_idx in kf.split(data_y):
        train_x = data_x.iloc[train_idx]
        train_y = data_y.iloc[train_idx]
        test_x = data_x.iloc[test_idx]
        test_y = data_y.iloc[test_idx]
        train_result, test_result = single_set_evaluation(train_x, test_x, train_y, test_y,
                                                          learners, evaluation_methods)
        test_fold_score.append(test_result)
        train_fold_score.append(train_result)
    return train_fold_score, test_fold_score


def learning_curve(data_x, data_y, learner, test_ratio: list, evaluation_methods=const.f1, shuffle=True):
    """
    test_ratio is a list of different test ratio to generate different # of testing cases vs training cases
    """
    learner = [learner]  # to fit single_set_evaluation function format
    test_ratio = sorted(test_ratio, reverse=True)  # we want less training cases at the beginning
    data_length = data_x.shape[0]
    idx = np.arange(data_length)
    test_fold_score = []
    train_fold_score = []
    if shuffle:
        np.random.shuffle(idx)

    for tr in test_ratio:
        train_length = int(np.round(data_length*(1-tr)))
        train_idx = idx[: train_length]
        test_idx = idx[train_length:]
        train_x = data_x.iloc[train_idx]
        # try:
        #     train_y = data_y.iloc[train_idx]
        # except:
        #     pass
        train_y = data_y.iloc[train_idx]
        test_x = data_x.iloc[test_idx]
        test_y = data_y.iloc[test_idx]
        # start = time.time()
        train_result, test_result = single_set_evaluation(train_x, test_x, train_y, test_y,
                                                          learner, evaluation_methods)
        # print(f"{type(learner[0])}: {time.time()-start}")
        test_fold_score.append(test_result)
        train_fold_score.append(train_result)
    return train_fold_score, test_fold_score


def avg_score(train_score: list, test_score: list):
    """
    input is 2 lists, train list and test list, each element is a dict, which is a fold (evaluation) result

    each method has avg train score, test score among folds
    return 2 dicts, key is method, value is 1 list, a list
    """
    fold_num = len(train_score)
    avg_train = {}
    avg_test = {}
    evaluation_methods = train_score[0].keys()

    for method in evaluation_methods:
        train_learners_score_list = []
        test_learners_score_list = []
        for i in range(fold_num):
            train_learners_score_list.append(train_score[i][method])
            test_learners_score_list.append(test_score[i][method])
        train_avg = np.average(train_learners_score_list, axis=0)
        test_avg = np.average(test_learners_score_list, axis=0)
        avg_train[method] = train_avg
        avg_test[method] = test_avg

    return avg_train, avg_test


def learner_compare_graphing(train_score, test_score, x_list, evaluation_methods, topic):
    avg_train, avg_test = avg_score(train_score, test_score)

    consolidate = {}
    for eva_method in evaluation_methods:
        train_result = avg_train[eva_method]
        test_result = avg_test[eva_method]
        both_result = np.array([train_result, test_result])
        both_result = pd.DataFrame(both_result.transpose(), columns=[const.train, const.test])
        consolidate[eva_method] = both_result
    plot_result(consolidate, topic, x_list, [const.train, const.test])


def plot_result(scores, topic, x_points=None, label=None, save=True):

    for key, value in scores.items():
        ploting_individual_method(value, topic + f'_{key}', x_points, label, save)


def ploting_individual_method(scores, title, x_points=None, label=None, save=True):
    if x_points is None:
        x_points = np.arrange(len(scores))
    plt.plot(x_points, scores)
    plt.legend(label)
    plt.title(title)
    if save:
        plt.savefig(title + ".png")
    else:
        plt.show()
    plt.close()


def build_learning_score(scores, evaluation_methods):
    """
    consolidate to {method1: score list, method2: score list}
    """
    consolidate_dict = {}
    for method in evaluation_methods:
        consolidate_dict[method] = []
    for score in scores:
        for method in evaluation_methods:
            consolidate_dict[method].append(score[method][0])

    return consolidate_dict


def learning_curve_graphing(train_score, test_score, x_list, evaluation_methods, topic):
    consolidate = {}
    for eva_method in evaluation_methods:
        train_result = train_score[eva_method]
        test_result = test_score[eva_method]
        both_result = np.array([train_result, test_result])
        both_result = pd.DataFrame(both_result.transpose(), columns=[const.train, const.test])
        consolidate[eva_method] = both_result
    plot_result(consolidate, topic, x_list, [const.train, const.test])
