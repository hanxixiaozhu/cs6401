from sklearn.cluster import KMeans
import data_generation as dg
import numpy as np
import config
import utils
import const

# def knn_clustering(data, k=10):
#     k_mean_cluster = KMeans(n_clusters=k).fit(data)
#     cluster_label = k_mean_cluster.labels_
#     return cluster_label


def knn_experiment(x_train, y_train, x_test, y_test, cluster_nums, return_accuracy_score=False, graph=False,
                   title=None):

    # cluster_nums = range(2, 20, 2)
    cluster_collections = []
    for num_cluster in cluster_nums:
        cluster_i = KMeans(n_clusters=num_cluster)
        cluster_collections.append(cluster_i)
    si_score_collection, mi_collection, accuracy_collection = \
        utils.clustering_experiment(x_train, y_train, x_test, y_test, cluster_collections, return_accuracy_score)
    if graph:
        scores = [si_score_collection, mi_collection, accuracy_collection]
        labels = [const.silhouette_score, const.adjusted_mutual_info_score, const.accuracy_score]
        # if return_accuracy_score:
        #     scores = [si_score_collection, mi_collection, accuracy_collection]
        #     labels = [const.silhouette_score, const.adjusted_mutual_info_score, const.accuracy_score]
        # else:
        #     scores = [si_score_collection, mi_collection]
        #     labels = [const.silhouette_score, const.adjusted_mutual_info_score]
        utils.experiment_graph(scores, title,
                               x_points=cluster_nums, label=labels)

        print(f"when the number of cluster is {cluster_nums[int (np.argmax(si_score_collection))]}, "
              f"silhouette_score is the highest with "
              f"{np.max(si_score_collection)}; \nwhen the number of cluster is "
              f"{cluster_nums[int (np.argmax(mi_collection))]},"
              f"the mutual information is the highest with "
              f"{np.max(mi_collection)}")
    return si_score_collection, mi_collection, accuracy_collection


def digit_data_knn_experiment():
    x_train, y_train, x_test, y_test = dg.digit_data_reading()
    cluster_nums = range(2, 20, 2)
    si_score_collection, mi_collection, accuracy_collection = \
        knn_experiment(x_train, y_train, x_test, y_test, cluster_nums, return_accuracy_score=False, graph=True,
                       title=const.digit_data_knn_silhouette_mi_score)

    return si_score_collection, mi_collection, accuracy_collection


def trading_data_knn_experiment():
    x_train, y_train, x_test, y_test = dg.trading_data_reading()
    cluster_nums = range(2, 5, 1)
    si_score_collection, mi_collection, accuracy_collection = \
        knn_experiment(x_train, y_train, x_test, y_test, cluster_nums, return_accuracy_score=True, graph=True,
                       title=const.trading_data_knn_silhouette_mi_score)
    return si_score_collection, mi_collection, accuracy_collection


def experiment_1():
    si, mi, acc = digit_data_knn_experiment()
    si2, mi2, acc2 = trading_data_knn_experiment()
    return si, mi, acc, si2, mi2, acc2


if __name__ == '__main__':
    np.random.seed(config.random_seed)
    experiment_1()
