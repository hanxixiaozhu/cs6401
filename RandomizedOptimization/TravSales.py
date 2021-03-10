import mlrose_hiive
import numpy as np
import utils
import config
from functools import partial


def problem_generation():
    """
    problem from tutorial example
    """
    dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), (0, 5, 5.3852),
                 (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), (1, 3, 2.8284), (1, 4, 2.0000),
                 (1, 5, 4.1231), (1, 6, 4.2426), (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361),
                 (2, 5, 4.4721), (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056),
                 (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), (4, 7, 2.2361),
                 (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
    fitness_dists = mlrose_hiive.TravellingSales(distances=dist_list)
    problem = mlrose_hiive.TSPOpt(length=8, fitness_fn=fitness_dists, maximize=False)
    return problem


def best_algo_experiment():
    """
    use tutorial problem
    """
    problem = problem_generation()
    utils.experiment_graphing(problem, None, 5, None, 'TSP_BestAlgoCurve.png', maximum=False, mimic_iteration=10)


if __name__ == '__main__':
    np.random.seed(config.seed)
    best_algo_experiment()
    # problem = problem_generation()
    # utils.approximate_time_experiment(problem, 10, config.seed, 300)
