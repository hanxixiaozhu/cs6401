import PCA
import ICA
import RandomProjections
import data_generation as dg
import l1_regularization
import utils
import const


def digit_experiments_part2():
    xt, yt, xtt, ytt = dg.digit_data_reading()
    num_components = range(1, xt.shape[1]+1)
    pca_exp_error = PCA.digit_data_pca_apply()
    ica_exp_error = ICA.digit_data_ica_apply()
    rp_exp_error = RandomProjections.digit_data_rp_apply()
    errors = [pca_exp_error, ica_exp_error, rp_exp_error]
    labels = [const.PCA, const.ICA, const.RandomProject]
    utils.experiment_graph(errors, const.reconstruction_errors + '_DigitData', x_points=num_components, label=labels)

    x, y = utils.train_test_combine(xt, yt, xtt, ytt)
    avg_kurtosis_collection = ICA.avg_kurtosis_ica(x)
    utils.experiment_graph([avg_kurtosis_collection], const.kurtosis + '_DigitData',
                           x_points=num_components, label=[const.kurtosis])


def trading_experiments_part2():
    xt, yt, xtt, ytt = dg.trading_data_reading()
    num_components = range(1, xt.shape[1]+1)
    pca_exp_error = PCA.trading_data_pca_apply()
    ica_exp_error = ICA.trading_data_ica_apply()
    rp_exp_error = RandomProjections.trading_data_rp_apply()
    errors = [pca_exp_error, ica_exp_error, rp_exp_error]
    labels = [const.PCA, const.ICA, const.RandomProject]
    utils.experiment_graph(errors, const.reconstruction_errors + '_TradingData', x_points=num_components, label=labels)

    x, y = utils.train_test_combine(xt, yt, xtt, ytt)
    avg_kurtosis_collection = ICA.avg_kurtosis_ica(x)
    utils.experiment_graph([avg_kurtosis_collection], const.kurtosis + '_TradingData',
                           x_points=num_components, label=[const.kurtosis])


def l1_regularization_experiment_part2():
    dx, dxi = l1_regularization.digit_x_selection()
    tx, txi = l1_regularization.trade_x_selection()
    return dx, dxi, tx, txi


if __name__ == '__main__':
    digit_experiments_part2()
    trading_experiments_part2()
