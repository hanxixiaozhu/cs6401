from sklearn.neural_network import MLPClassifier
import data_generation as dg
import numpy as np
import utils
from sklearn.metrics import f1_score, roc_auc_score
import const


def trading_bmk_exp():
    testing_ratio = np.arange(0.1, 1, 0.05)
    f1_scores_collections = []
    roc_scores_collections = []

    trading_x_train, trading_y_train, trading_x_test, trading_y_test = dg.trading_data_reading()

    nn = MLPClassifier((64, 64, 64))
    for tr in testing_ratio:
        x_train, x_test, y_train, y_test = utils.train_test_split(trading_x_test, trading_y_test, test_size=tr,
                                                                  stratify=trading_y_test)
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

    labels = ['bmk']
    utils.experiment_graph([f1_scores_collections], const.trading_data_bmk_f1, x_points=testing_ratio, label=labels)
    utils.experiment_graph([roc_scores_collections], const.trading_data_bmk_roc, x_points=testing_ratio, label=labels)


if __name__ == '__main__':
    trading_bmk_exp()
