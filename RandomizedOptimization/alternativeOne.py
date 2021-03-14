import mlrose_hiive
import numpy as np
import utils
import config


def best_algo_experiment_easy():
    """
    run for simulation with the best hyperparam
    """
    problem = mlrose_hiive.FlipFlopOpt(length=20)
    utils.experiment_graphing(problem, None, 5, None, 'AlternativeOne_20_Exp_BestAlgoCurve.png')


def best_algo_experiment_hard():
    """
    run for simulation with the best hyperparam
    """
    problem = mlrose_hiive.FlipFlopOpt(length=500)
    utils.experiment_graphing(problem, None, 5, None, 'AlternativeOneExp_BestAlgoCurve.png')


if __name__ == '__main__':
    np.random.seed(config.seed)
    best_algo_experiment_hard()
    best_algo_experiment_easy()
