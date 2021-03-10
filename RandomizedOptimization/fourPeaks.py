import mlrose_hiive
import numpy as np
import utils
from functools import partial
import config


def best_algo_experiment():
    """
    run for simulation with the best hyperparam
    """
    problem = mlrose_hiive.DiscreteOpt(length=500, fitness_fn=mlrose_hiive.FourPeaks(), maximize=True, max_val=2)
    utils.experiment_graphing(problem, None, 5, None, 'FourPeaksExp_BestAlgoCurve.png')


if __name__ == '__main__':
    np.random.seed(config.seed)
    best_algo_experiment()
