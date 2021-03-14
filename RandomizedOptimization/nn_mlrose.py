from sklearn import datasets
import mlrose_hiive
import numpy as np
import config
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import time
import utils


def split_data(data_x, data_y, train_ratio):
    # data_x, data_y = datasets.load_digits(return_X_y=True, as_frame=True)
    data_length = data_x.shape[0]
    data_idx = np.arange(data_length)
    np.random.shuffle(data_idx)
    data_train_length = int(data_length * train_ratio)
    train_idx = data_idx[: data_train_length]
    test_idx = data_idx[data_train_length:]

    data_x_train = data_x.iloc[train_idx]
    data_y_train = data_y.iloc[train_idx]

    data_x_test = data_x.iloc[test_idx]
    data_y_test = data_y.iloc[test_idx]
    return data_x_train.reset_index(drop=True), data_y_train.reset_index(drop=True), \
        data_x_test.reset_index(drop=True), data_y_test.reset_index(drop=True)


def mlrose_train_test_experiment(train_x, train_y, test_x, test_y, learner):
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(np.array(train_y).reshape((-1, 1))).todense()
    # y_test_hot = one_hot.transform(np.array(test_y).reshape(-1, 1)).todense()
    learner.fit(train_x, y_train_hot)
    test_result = learner.predict(test_x)
    test_score = learner.predicted_probs
    result_nothot = np.argmax(test_result, axis=1)
    train_result = learner.predict(train_x)
    train_result_nothot = np.argmax(train_result, axis=1)

    f1_train = metrics.f1_score(train_y, train_result_nothot, average='weighted')
    f1_test = metrics.f1_score(test_y, result_nothot, average='weighted')

    roc_test = metrics.roc_auc_score(test_y, test_score, average='weighted', multi_class='ovr')
    confusion_matrix_test = metrics.confusion_matrix(test_y, result_nothot)

    accuracy_train = metrics.accuracy_score(train_y, train_result_nothot)
    accuracy_test = metrics.accuracy_score(test_y, result_nothot)

    curve = learner.fitness_curve

    return f1_train, roc_test, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve


def sklearn_mlp_experiment(train_x, train_y, test_x, test_y):
    learner = MLPClassifier(hidden_layer_sizes=(25, 25))
    learner.fit(train_x, train_y)
    train_result = learner.predict(train_x)
    test_result = learner.predict(test_x)
    f1_train = metrics.f1_score(train_y, train_result, average='macro')
    f1_test = metrics.f1_score(test_y, test_result, average='macro')

    confusion_matrix_train = metrics.confusion_matrix(train_y, train_result)
    confusion_matrix_test = metrics.confusion_matrix(test_y, test_result)

    accuracy_train = metrics.accuracy_score(train_y, train_result)
    accuracy_test = metrics.accuracy_score(test_y, test_result)
    return f1_train, confusion_matrix_train, f1_test, confusion_matrix_test, accuracy_train, accuracy_test


def mlrose_nn_rhc(train_x, train_y, test_x, test_y):
    rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[25, 25], curve=True, max_iters=10000)
    f1_train, roc_test, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve \
        = mlrose_train_test_experiment(train_x, train_y, test_x, test_y, rhc_nn)
    return f1_train, roc_test, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve


def mlrose_nn_sa(train_x, train_y, test_x, test_y):
    sa_nn = mlrose_hiive.NeuralNetwork(algorithm='simulated_annealing', hidden_nodes=[25, 25], curve=True,
                                       max_iters=10000)
    f1_train, roc_test, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve \
        = mlrose_train_test_experiment(train_x, train_y, test_x, test_y, sa_nn)
    return f1_train, roc_test, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve


def mlrose_nn_ga(train_x, train_y, test_x, test_y):
    ga_nn = mlrose_hiive.NeuralNetwork(algorithm='genetic_alg', hidden_nodes=[25, 25], curve=True, max_iters=10000)
    f1_train, roc_test, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve \
        = mlrose_train_test_experiment(train_x, train_y, test_x, test_y, ga_nn)
    return f1_train, roc_test, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve


def nn_default_experiment(train_x, train_y, test_x, test_y):
    mlp_start = time.time()
    result_mlp = sklearn_mlp_experiment(train_x, train_y, test_x, test_y)
    mlp_str = f"mlp nn takes {time.time() - mlp_start}"
    utils.recorder(mlp_str, "mlp_nn_time")
    print(mlp_str)

    rhc_start = time.time()
    result_rhc = mlrose_nn_rhc(train_x, train_y, test_x, test_y)
    rhc_str = f"rhc nn takes {time.time() - rhc_start}"
    utils.recorder(rhc_str, "rhc_nn_time")
    print(rhc_str)

    sa_start = time.time()
    result_sa = mlrose_nn_sa(train_x, train_y, test_x, test_y)
    sa_str = f"sa nn takes {time.time() - sa_start}"
    utils.recorder(sa_str, "sa_nn_time")
    print(sa_str)

    ga_start = time.time()
    result_ga = mlrose_nn_ga(train_x, train_y, test_x, test_y)
    ga_str = f"ga nn takes {time.time() - ga_start}"
    utils.recorder(ga_str, "ga_nn_time")
    print(ga_str)
    return result_mlp, result_rhc, result_sa, result_ga


def gd_best_pick(train_x, train_y, test_x, test_y, num_iter=10):
    iterations = [[i] for i in range(num_iter)]
    result_list = []
    time_collection = []

    for _ in iterations:
        print(_)
        start = time.time()
        learner = mlrose_hiive.NeuralNetwork(algorithm='gradient_descent', hidden_nodes=[25, 25], curve=True,
                                             max_iters=1000)
        result_list.append(mlrose_train_test_experiment(train_x, train_y, test_x, test_y, learner))
        time_collection.append(time.time() - start)
    utils.nn_result_experiment(result_list, iterations, 'GD', ['Iteration'],
                               np.mean(time_collection))
    return result_list, np.mean(time_collection)


def rhc_best_pick(train_x, train_y, test_x, test_y):
    restarts = list(range(1, 10))
    max_attempts = list(range(10, 100, 10))
    key_arg = []
    for re in restarts:
        for att in max_attempts:
            key_arg.append([re, att])
    result_list = []
    time_collection = []
    itera = 0
    for arg in key_arg:
        print(itera)
        itera += 1
        start = time.time()
        learner = mlrose_hiive.NeuralNetwork(algorithm='random_hill_climb',
                                             restarts=arg[0],
                                             max_attempts=arg[1],
                                             max_iters=5000,
                                             learning_rate=0.001, curve=True)
        result_list.append(mlrose_train_test_experiment(train_x, train_y, test_x, test_y, learner))
        time_collection.append(time.time() - start)
    utils.nn_result_experiment(result_list, key_arg, 'RHC', ['Restart', 'Max Attempts'],
                               np.mean(time_collection))
    return result_list, np.mean(time_collection)


def sa_best_pick(train_x, train_y, test_x, test_y):
    initial_temperature = np.arange(1, 51, 5)
    scheduler_list = []
    key_arg = []
    for ini_t in initial_temperature:
        schedule1 = mlrose_hiive.GeomDecay(init_temp=ini_t)
        schedule2 = mlrose_hiive.ExpDecay(init_temp=ini_t)
        scheduler_list.append(schedule1)
        scheduler_list.append(schedule2)
        key_arg.append([ini_t, 'GeoDecay'])
        key_arg.append([ini_t, 'ExpDecay'])
    result_list = []
    time_collection = []
    for scheduler in scheduler_list:
        start = time.time()
        learner = mlrose_hiive.NeuralNetwork(algorithm='random_hill_climb', schedule=scheduler, max_iters=10000,
                                             learning_rate=0.001, curve=True)
        result_list.append(mlrose_train_test_experiment(train_x, train_y, test_x, test_y, learner))
        time_collection.append(time.time() - start)
    utils.nn_result_experiment(result_list, key_arg, 'SA', ['Initial Temperature', 'Decay Schedule'],
                               np.mean(time_collection))
    return result_list, np.mean(time_collection)


def ga_best_pick(train_x, train_y, test_x, test_y):
    pop_size_list = list(range(100, 300, 50))
    mutation_prob_list = np.arange(0.05, 0.25, 0.05)
    key_arg = []
    for pop in pop_size_list:
        for mut in mutation_prob_list:
            key_arg.append((pop, mut))
    result_list = []
    time_collection = []
    for arg in key_arg:
        start = time.time()
        learner = mlrose_hiive.NeuralNetwork(algorithm='genetic_alg',
                                             pop_size=arg[0], mutation_prob=arg[1], max_iters=5000,
                                             learning_rate=0.001, curve=True)
        result_list.append(mlrose_train_test_experiment(train_x, train_y, test_x, test_y, learner))
        time_collection.append(time.time() - start)
    utils.nn_result_experiment(result_list, key_arg, 'GA', ['Population Size', 'Mutation Rate'],
                               np.mean(time_collection))
    return result_list, np.mean(time_collection)


if __name__ == '__main__':
    np.random.seed(config.seed)
    data_x, data_y = datasets.load_digits(return_X_y=True, as_frame=True)
    digit_x_train, digit_y_train, digit_x_test, digit_y_test = split_data(data_x, data_y, config.train_data_ratio)
    # mlp_r, rhc_r, sa_r, ga_r = nn_default_experiment(digit_x_train, digit_y_train, digit_x_test, digit_y_test)
    rhc_best_r, rhc_avg_time = rhc_best_pick(digit_x_train, digit_y_train, digit_x_test, digit_y_test)
    sa_best_r, sa_avg_time = sa_best_pick(digit_x_train, digit_y_train, digit_x_test, digit_y_test)
    ga_best_r, ga_avg_time = ga_best_pick(digit_x_train, digit_y_train, digit_x_test, digit_y_test)
