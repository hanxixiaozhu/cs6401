import mlrose_hiive
import numpy as np
import utils
import config
import itertools


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
    the_minimal_cost(dist_list, 8)
    return problem


def the_minimal_cost(dist_list, num_node):
    # dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), (0, 5, 5.3852),
    #              (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), (1, 3, 2.8284), (1, 4, 2.0000),
    #              (1, 5, 4.1231), (1, 6, 4.2426), (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361),
    #              (2, 5, 4.4721), (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056),
    #              (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), (4, 7, 2.2361),
    #              (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
    all_paths = list(itertools.permutations(np.arange(num_node)))
    dist_p2 = [(x[1], x[0], x[2]) for x in dist_list]
    all_dist = set(dist_list).union(set(dist_p2))
    a = {}
    for dist in all_dist:
        a[(dist[0], dist[1])] = dist[2]
    costs = []
    for p in all_paths:
        co = 0
        for i in range(len(p) - 1):
            co += a[(p[i], p[i + 1])]
        costs.append(co)
    print(f"optimum (minimum) cost of {dist_list} is {min(costs)}")
    return min(costs)


def best_algo_experiment_easy():
    """
    use tutorial problem
    """
    problem = problem_generation()
    utils.experiment_graphing(problem, None, 5, None, 'TSP_BestAlgoCurve.png', maximum=False, mimic_iteration=10)


def best_algo_experiment_hard():
    """
    use tutorial problem
    """
    num_nodes = 20
    dist_list = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist_list.append((i, j, np.random.uniform(0, 5)))

    fitness_dists = mlrose_hiive.TravellingSales(distances=dist_list)
    problem = mlrose_hiive.TSPOpt(length=num_nodes, fitness_fn=fitness_dists, maximize=False)
    # the_minimal_cost(dist_list, 20) too complicated
    utils.experiment_graphing(problem, None, 10, None, f'TSP_{num_nodes}_nodes_BestAlgoCurve.png', maximum=False,
                              mimic_iteration=1)


if __name__ == '__main__':
    np.random.seed(config.seed)
    # the_minimal_cost()
    best_algo_experiment_easy()
    best_algo_experiment_hard()
    # problem = problem_generation()
    # utils.approximate_time_experiment(problem, 10, config.seed, 300)
