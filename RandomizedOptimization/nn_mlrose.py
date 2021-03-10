from sklearn import datasets
import mlrose_hiive
import numpy as np
import config
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics


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
    result_nothot = np.argmax(test_result, axis=1)
    train_result = learner.predict(train_x)
    train_result_nothot = np.argmax(train_result, axis=1)

    f1_train = metrics.f1_score(train_y, train_result_nothot, average='macro')
    f1_test = metrics.f1_score(test_y, result_nothot, average='macro')

    confusion_matrix_train = metrics.confusion_matrix(train_y, train_result_nothot)
    confusion_matrix_test = metrics.confusion_matrix(test_y, result_nothot)

    accuracy_train = metrics.accuracy_score(train_y, train_result_nothot)
    accuracy_test = metrics.accuracy_score(test_y, result_nothot)

    curve = learner.fitness_curve
    return f1_train, confusion_matrix_train, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve


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
    rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[25, 25], curve=True, max_iters=1000)
    f1_train, confusion_matrix_train, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve \
        = mlrose_train_test_experiment(train_x, train_y, test_x, test_y, rhc_nn)
    return f1_train, confusion_matrix_train, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve


def mlrose_nn_sa(train_x, train_y, test_x, test_y):
    rhc_nn = mlrose_hiive.NeuralNetwork(algorithm='simulated_annealing', hidden_nodes=[25, 25], curve=True,
                                        max_iters=1000)
    f1_train, confusion_matrix_train, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve \
        = mlrose_train_test_experiment(train_x, train_y, test_x, test_y, rhc_nn)
    return f1_train, confusion_matrix_train, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve


def mlrose_nn_ga(train_x, train_y, test_x, test_y):
    rhc_nn = mlrose_hiive.NeuralNetwork(algorithm='genetic_alg', hidden_nodes=[25, 25], curve=True, max_iters=1000)
    f1_train, confusion_matrix_train, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve \
        = mlrose_train_test_experiment(train_x, train_y, test_x, test_y, rhc_nn)
    return f1_train, confusion_matrix_train, f1_test, confusion_matrix_test, accuracy_train, accuracy_test, curve


if __name__ == '__main__':
    np.random.seed(config.seed)
    data_x, data_y = datasets.load_digits(return_X_y=True, as_frame=True)
    digit_x_train, digit_y_train, digit_x_test, digit_y_test = split_data(data_x, data_y, config.train_data_ratio)
    result_mlp = sklearn_mlp_experiment(digit_x_train, digit_y_train, digit_x_test, digit_y_test)
    result_rhc = mlrose_nn_rhc(digit_x_train, digit_y_train, digit_x_test, digit_y_test)
    result_sa = mlrose_nn_sa(digit_x_train, digit_y_train, digit_x_test, digit_y_test)
    result_ga = mlrose_nn_ga(digit_x_train, digit_y_train, digit_x_test, digit_y_test)
    # one_hot = OneHotEncoder()
    # digit_y_train_hot = one_hot.fit_transform(np.array(digit_y_train).reshape((-1, 1))).todense()
    # digit_y_test_hot = one_hot.transform(np.array(digit_y_test).reshape(-1, 1)).todense()
    # lr = mlrose_hiive.NeuralNetwork(curve=True)
    # lr.fit(digit_x_train, digit_y_train_hot)
    # result_test = lr.predict(digit_x_test)
    # result_nothot = np.argmax(result_test, axis=1)
    # f1 = metrics.f1_score(digit_y_test, result_nothot, average='macro')
    # confusion_matrix = metrics.confusion_matrix(digit_y_test, result_nothot)
