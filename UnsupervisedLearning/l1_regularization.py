from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import data_generation as dg
import numpy as np


def l1_logistic_selection(x_train, y_train):
    logistic = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear')
    trained_logistic = logistic.fit(x_train, y_train)
    model = SelectFromModel(trained_logistic, prefit=True)
    x_selected = model.transform(x_train)
    x_mask = np.array(model.get_support())
    selected_x_indice = [i for i in range(len(x_mask)) if x_mask[i]]
    return x_selected, selected_x_indice


def digit_x_selection():
    digit_x_train, digit_y_train, digit_x_test, digit_y_test = dg.digit_data_reading()
    return l1_logistic_selection(digit_x_train, digit_y_train)


def trade_x_selection():
    trading_x_train, trading_y_train, trading_x_test, trading_y_test = dg.trading_data_reading()
    return l1_logistic_selection(trading_x_train, trading_y_train)


if __name__ == '__main__':
    x, xi = trade_x_selection()
