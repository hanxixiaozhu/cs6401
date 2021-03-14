import mlrose_hiive
import numpy as np
import utils
import config


def best_algo_experiment_easy():
    """
    run for simulation with the best hyperparam
    """
    problem = mlrose_hiive.DiscreteOpt(length=50, fitness_fn=mlrose_hiive.FourPeaks(), maximize=True, max_val=2)
    utils.experiment_graphing(problem, None, 5, None, 'FourPeaksExp_50_BestAlgoCurve.png')


def best_algo_experiment_hard():
    """
    run for simulation with the best hyperparam
    """
    problem = mlrose_hiive.DiscreteOpt(length=500, fitness_fn=mlrose_hiive.FourPeaks(), maximize=True, max_val=2)
    utils.experiment_graphing(problem, None, 5, None, 'FourPeaksExp_BestAlgoCurve.png')


def rhc_heuristic():
    problem_hard = mlrose_hiive.DiscreteOpt(length=500, fitness_fn=mlrose_hiive.FourPeaks(t_pct=0.1),
                                            maximize=True, max_val=2)
    problem_easy = mlrose_hiive.DiscreteOpt(length=500, fitness_fn=mlrose_hiive.FourPeaks(t_pct=0.002),
                                            maximize=True, max_val=2)
    result_hard = mlrose_hiive.random_hill_climb(problem_hard, max_attempts=500, restarts=100, curve=True)
    result_easy = mlrose_hiive.random_hill_climb(problem_easy, max_attempts=500, restarts=100, curve=True)
    print(f"hard problem result score {result_hard[1]}, easy problem result score {result_easy[1]}")
    return result_hard, result_easy


if __name__ == '__main__':
    np.random.seed(config.seed)
    best_algo_experiment_hard()
    best_algo_experiment_easy()
    res_hard, res_easy = rhc_heuristic()
