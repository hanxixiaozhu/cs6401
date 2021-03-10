import mlrose_hiive
import numpy as np
import utils
import config
from functools import partial


def best_algo_experiment():
    """
    run for simulation with the best hyperparam
    """
    problem = mlrose_hiive.FlipFlopOpt(length=500)
    utils.experiment_graphing(problem, None, 5, None, 'AlternativeOneExp_BestAlgoCurve.png')


# def sa_pick(problem)
if __name__ == '__main__':
    np.random.seed(config.seed)
    best_algo_experiment()
