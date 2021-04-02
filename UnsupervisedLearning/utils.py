from sklearn.metrics import silhouette_score, accuracy_score, adjusted_mutual_info_score, mean_squared_error, \
    f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from itertools import permutations
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import l1_regularization
import const
import statistics


def train_test_combine(x_train, y_train, x_test, y_test):
    if not isinstance(x_train, pd.DataFrame):
        x_train = pd.DataFrame(x_train)
    x = pd.concat([x_train, x_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)
    return x, y


# def train_test_split(x, y, test_size):
#     pass


def experiment_graph(scores, title, x_points=None, label=None, save=True):
    if x_points is None:
        x_points = np.arrange(len(scores))
    for score in scores:
        plt.plot(x_points, score)
    plt.legend(label)
    plt.title(title)
    if save:
        plt.savefig(title + ".png")
    else:
        plt.show()
    plt.close()


"""
Clustering utils
"""


def find_max_accuracy_score(y_true, y_predict):
    true_class_num = len(set(y_true))
    train_class_num = len(set(y_predict))
    total_class = max(train_class_num, true_class_num)
    class_list = list(range(total_class))
    permute = list(permutations(class_list))
    mapping_dict_list = [dict(zip(class_list, x)) for x in permute]
    max_score = -1
    for possible_mapping in mapping_dict_list:
        mapped_y = np.vectorize(possible_mapping.get)(y_predict)
        acc_score = accuracy_score(y_true, mapped_y)
        max_score = max(max_score, acc_score)
    return max_score


def clustering_experiment(x_train, y_train, x_test, y_test, clusters, return_accuracy_score=False):
    """
    clusters: list of clusters
    """
    x, y = train_test_combine(x_train, y_train, x_test, y_test)
    si_scores = []
    mi_score = []
    acc_score = []
    for cl in clusters:
        print(cl)
        start = time.time()
        cl_fit = cl.fit(x)
        # try:
        #     labels = cl_fit.labels_
        # except AttributeError:
        labels = cl_fit.predict(x)
        silhou = silhouette_score(x, labels)
        print("it takes ")
        print(time.time()-start)
        mi = adjusted_mutual_info_score(y, labels)
        si_scores.append(silhou)
        mi_score.append(mi)
        if return_accuracy_score:
            acc_sc = find_max_accuracy_score(y, labels)
        else:
            possible_labels = set(y)
            acc_sc = 0
            len_y = len(y)
            for label in possible_labels:
                group_l = labels[np.where(y == label)]
                # len_group = len(group_l)
                try:
                    mode_len = len(np.where(group_l == statistics.mode(group_l))[0])
                except statistics.StatisticsError:
                    any_model = Counter(group_l).most_common(1)[0][0]
                    mode_len = len(np.where(group_l == any_model)[0])
                acc_sc += mode_len/len_y
        acc_score.append(acc_sc)
    return si_scores, mi_score, acc_score


"""
part 2
"""


def one_dr_apply(x, dr, return_type=const.mean_error):
    # transform is x*matrix
    x_dr = dr.fit_transform(x)
    # x_revert is x_dr * matrix^-1
    if return_type == const.mean_error:
        revert_x = np.dot(x_dr, np.linalg.pinv(dr.components_.T))
        mean_error = mean_squared_error(x, revert_x)
        return mean_error
    elif return_type == const.eigenvalues:
        return dr.explained_variance_
    elif return_type == const.eigenvalues_ratio:
        return dr.explained_variance_ratio_


def dr_apply_experiment(x, drs, return_type=const.mean_error):
    mean_errors = []
    for dr in drs:
        mr = one_dr_apply(x, dr, return_type)
        mean_errors.append(mr)
    return mean_errors


"""
part 3
"""


def x_preprocess(x_train, x_test):
    x = pd.concat([x_train, x_test], axis=0)
    x = StandardScaler().fit_transform(x)
    return x


"""
part 4
"""


def gen_x_train_preprocessor(x_train, dr):
    standardizer = StandardScaler().fit(x_train)
    x_starndardized = standardizer.transform(x_train)
    dr_fit = dr.fit(x_starndardized)
    x_reducted = dr_fit.transform(x_starndardized)
    return standardizer, dr_fit, x_reducted


def single_dr_nn_experiment(x_train, y_train, x_test, dr, nn):
    standardizer, dr_fit, x_train_reducted = gen_x_train_preprocessor(x_train, dr)
    nn.fit(x_train_reducted, y_train)
    x_test_standardized = standardizer.transform(x_test)
    x_test_reducted = dr_fit.transform(x_test_standardized)
    y_predict = nn.predict(x_test_reducted)
    y_predict_prob = nn.predict_proba(x_test_reducted)
    return y_predict, y_predict_prob


def grid_compare_util(x, y, validation_size, dr, nn_collections):
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=validation_size)
    y_predict_collection = []
    y_prob_predict_collection = []
    for nn in nn_collections:
        y_predict, y_predict_prob = single_dr_nn_experiment(x_train, y_train, x_valid, dr, nn)
        y_predict_collection.append(y_predict)
        y_prob_predict_collection.append(y_predict_prob)
    return y_predict_collection, y_prob_predict_collection, y_valid


def grid_compare(x, y, validation_size, dr, nn_collections):
    y_predict_collection, y_prob_predict_collection, y_valid = \
        grid_compare_util(x, y, validation_size, dr, nn_collections)
    f1_scores_collections = []
    roc_scores_collections = []
    for i in range(len(y_predict_collection)):
        fs = f1_score(y_valid, y_predict_collection[i], average='micro')
        if y_prob_predict_collection[i].shape[1] == 2:
            auc_roc = roc_auc_score(y_valid, y_prob_predict_collection[i][:, 1], multi_class='ovr')
        else:
            auc_roc = roc_auc_score(y_valid, y_prob_predict_collection[i], multi_class='ovr')
        f1_scores_collections.append(fs)
        roc_scores_collections.append(auc_roc)
    return f1_scores_collections, roc_scores_collections


def l1_nn_grid(x, y, validation_size, nn_collections):
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=validation_size)
    x_selected, selected_x_indice = l1_regularization.l1_logistic_selection(x_train, y_train)
    x_valid_select = x_valid.iloc[:, selected_x_indice]
    f1_scores_collections = []
    roc_scores_collections = []
    for nn in nn_collections:
        nn.fit(x_selected, y_train)
        y_predict = nn.predict(x_valid_select)
        y_predict_prob = nn.predict_proba(x_valid_select)
        fs = f1_score(y_valid, y_predict, average='micro')
        if y_predict_prob.shape[1] == 2:
            auc_roc = roc_auc_score(y_valid, y_predict_prob[:, 1], multi_class='ovr')
        else:
            auc_roc = roc_auc_score(y_valid, y_predict_prob, multi_class='ovr')
        f1_scores_collections.append(fs)
        roc_scores_collections.append(auc_roc)
    return f1_scores_collections, roc_scores_collections, selected_x_indice


def build_learning_curve(x, y, dr, nn):
    testing_ratio = np.arange(0.1, 1, 0.05)
    stder, dr_fit, x_reducted = gen_x_train_preprocessor(x, dr)
    f1_scores_collections = []
    roc_scores_collections = []
    for tr in testing_ratio:
        x_train, x_test, y_train, y_test = train_test_split(x_reducted, y, test_size=tr, stratify=y)
        nn.fit(x_train, y_train)
        y_preidct = nn.predict(x_test)
        y_prob_predict = nn.predict_proba(x_test)
        f1 = f1_score(y_test, y_preidct, average='micro')
        if y_prob_predict.shape[1] == 2:
            roc_score = roc_auc_score(y_test, y_prob_predict[:, 1], multi_class='ovr')
        else:
            try:
                roc_score = roc_auc_score(y_test, y_prob_predict, multi_class='ovr')
            except:
                p=1
        f1_scores_collections.append(f1)
        roc_scores_collections.append(roc_score)
    return f1_scores_collections, roc_scores_collections, testing_ratio


def build_learning_curve_l1(x, y, nn):
    testing_ratio = np.arange(0.1, 1, 0.05)
    f1_scores_collections = []
    roc_scores_collections = []
    for tr in testing_ratio:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=tr, stratify=y)
        nn.fit(x_train, y_train)
        y_preidct = nn.predict(x_test)
        y_prob_predict = nn.predict_proba(x_test)
        f1 = f1_score(y_test, y_preidct, average='micro')
        if y_prob_predict.shape[1] == 2:
            roc_score = roc_auc_score(y_test, y_prob_predict[:, 1], multi_class='ovr')
        else:
            roc_score = roc_auc_score(y_test, y_prob_predict, multi_class='ovr')
        f1_scores_collections.append(f1)
        roc_scores_collections.append(roc_score)
    return f1_scores_collections, roc_scores_collections, testing_ratio


"""
part 5
"""


def cluster_covert(cl, x_train, x_test):
    cl_fit = cl.fit(x_train)
    x_train_clustered = cl.predict(x_train)
    x_test_clustered = cl.predict(x_test)
    return cl_fit, x_train_clustered, x_test_clustered


def cluster_nn_grid(x, y, validation_size, cl, nn_collections):
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=validation_size)
    cl_fit, x_train_clustered, x_valid_clustered = cluster_covert(cl, x_train, x_valid)
    x_train_extra = x_train.copy()
    x_valid_extra = x_valid.copy()
    x_train_extra[const.cluster] = x_train_clustered
    x_valid_extra[const.cluster] = x_valid_clustered
    f1_scores_collections = []
    roc_scores_collections = []
    for nn in nn_collections:
        nn.fit(x_train_extra, y_train)
        y_predict = nn.predict(x_valid_extra)
        y_predict_prob = nn.predict_proba(x_valid_extra)
        fs = f1_score(y_valid, y_predict, average='micro')
        if y_predict_prob.shape[1] == 2:
            roc_score = roc_auc_score(y_valid, y_predict_prob[:, 1], multi_class='ovr')
        else:
            roc_score = roc_auc_score(y_valid, y_predict_prob, multi_class='ovr')
        f1_scores_collections.append(fs)
        roc_scores_collections.append(roc_score)
    return f1_scores_collections, roc_scores_collections


def cluster_learning_curve_build(x, y, cl, nn):
    x_label = cl.fit_predict(x)
    testing_ratio = np.arange(0.1, 1, 0.05)
    f1_scores_collections = []
    roc_scores_collections = []
    x_extra = x.copy()
    x_extra[const.cluster] = x_label
    for tr in testing_ratio:
        x_train, x_test, y_train, y_test = train_test_split(x_extra, y, test_size=tr, stratify=y)
        nn.fit(x_train, y_train)
        y_preidct = nn.predict(x_test)
        y_prob_predict = nn.predict_proba(x_test)
        f1 = f1_score(y_test, y_preidct, average='micro')
        if y_prob_predict.shape[1] == 2:
            roc_score = roc_auc_score(y_test, y_prob_predict[:, 1], multi_class='ovr')
        else:
            roc_score = roc_auc_score(y_test, y_prob_predict, multi_class='ovr')
        f1_scores_collections.append(f1)
        roc_scores_collections.append(roc_score)
    return f1_scores_collections, roc_scores_collections, testing_ratio
