import mlrose_hiive
import numpy as np
from functools import partial
import time
import const
import matplotlib.pyplot as plt
import pandas as pd


def single_solve(problem, solver):
    best_state, best_fitness, curve = solver(problem)
    return best_state, best_fitness, curve


def solver_simulation(problem, solver, iteration, seed):
    """
    return a collection of iteration
    """
    np.random.seed(seed)
    state_collection = []
    fitness_collection = []
    curve_collection = []
    time_taken = []
    for i in range(iteration):
        start_time = time.time()
        bs, bf, curve = single_solve(problem, solver)
        end_time = time.time()
        time_taken.append(end_time - start_time)
        state_collection.append(bs)
        fitness_collection.append(bf)
        curve_collection.append(curve)
    return state_collection, fitness_collection, curve_collection, time_taken


def solver_pick(problem, solver_list, iteration, seed, criteria=0, maximum=True):
    """
    criteria, the criteria to judge the best solver, 0 is best fitness, 1 is the shortest time, 2 will be both,
    and fitness will be the first criteria
    if same performance, will output all.
    """
    scores = []
    times = []
    curve_hist = []
    for solver in solver_list:
        # print(solver)
        bsc, bfc, curc, timec = solver_simulation(problem, solver, iteration, seed)
        avg_fit = np.mean(bfc)
        avg_time = np.mean(timec)
        scores.append(avg_fit)
        times.append(avg_time)
        curve_hist.append(curc[0])  # pick the first curve for simplicity
    if criteria == 0:
        if maximum:
            best_idx = np.argwhere(scores == np.amax(scores))
        else:
            best_idx = np.argwhere(scores == np.amin(scores))
    elif criteria == 1:
        best_idx = np.argwhere(times == np.amin(times))
        pass
    else:
        best_idx = np.argwhere(scores == np.amax(scores))
        pass
    best_idx = best_idx.flatten().tolist()
    best_solver = [solver_list[idx] for idx in best_idx]
    solver_score = [scores[idx] for idx in best_idx]
    solver_time = [times[idx] for idx in best_idx]
    best_curve_hist = [curve_hist[idx] for idx in best_idx]

    return best_solver, solver_score, solver_time, best_idx, best_curve_hist


def rhc_pick(problem, iteration, seed, criteria=0, maximum=True, curve=True):
    # max_attemps = list(range(5, 30, 5))
    max_attemps = np.arange(5, 30, 5)
    solver_list = []
    for ma in max_attemps:
        solver = partial(mlrose_hiive.random_hill_climb, max_attempts=ma.item(), curve=curve)
        solver_list.append(solver)
    best_rhc, best_score, best_time, best_idx, best_curve = \
        solver_pick(problem, solver_list, iteration, seed, criteria, maximum)
    return best_rhc, best_score, best_time, max_attemps[best_idx], best_curve


def sa_pick(problem, iteration, seed, criteria=0, maximum=True, curve=True):
    """
    pick for different initial temperature and
    """
    initial_temperature = np.arange(1, 51, 10)
    solver_list = []
    scheduler_list = []
    for ini_t in initial_temperature:
        schedule1 = mlrose_hiive.GeomDecay(init_temp=ini_t)
        schedule2 = mlrose_hiive.ExpDecay(init_temp=ini_t)
        scheduler_list.append(schedule1)
        scheduler_list.append(schedule2)
    for scheduler in scheduler_list:
        solver = partial(mlrose_hiive.simulated_annealing, schedule=scheduler, curve=curve)
        solver_list.append(solver)
    scheduler_list = np.array(scheduler_list)
    best_sa, best_score, best_time, best_idx, best_curve\
        = solver_pick(problem, solver_list, iteration, seed, criteria, maximum)
    return best_sa, best_score, best_time, scheduler_list[best_idx], best_curve


def ga_pick(problem, iteration, seed, criteria=0, maximum=True, curve=True):
    key_arg_list = []
    solver_list = []
    pop_breed_percent_list = np.arange(0.25, 0.9, 0.25)
    mutation_prob_list = np.arange(0.05, 0.25, 0.05)
    for pbp in pop_breed_percent_list:
        for mp in mutation_prob_list:
            key_arg = {const.pop_breed_percent: pbp, const.mutation_prob: mp}
            key_arg_list.append(key_arg)

    for key_arg in key_arg_list:
        solver = partial(mlrose_hiive.genetic_alg, pop_breed_percent=key_arg[const.pop_breed_percent],
                         mutation_prob=key_arg[const.mutation_prob], curve=curve)
        solver_list.append(solver)
    best_ga, best_score, best_time, best_idx, best_curve\
        = solver_pick(problem, solver_list, iteration, seed, criteria, maximum)
    key_arg_list = np.array(key_arg_list)
    return best_ga, best_score, best_time, key_arg_list[best_idx], best_curve


def mimic_pick(problem, iteration, seed, criteria=0, maximum=True, curve=True):
    key_arg_list = []
    solver_list = []
    pop_size_list = [200]  # use 200 as it is default value
    keep_pct_list = [0.2]  # use 0.2 as it is default value
    for ps in pop_size_list:
        for kp in keep_pct_list:
            key_arg = {const.pop_size: ps, const.keep_pct: kp}
            key_arg_list.append(key_arg)

    for key_arg in key_arg_list:
        solver = partial(mlrose_hiive.mimic, pop_size=key_arg[const.pop_size], keep_pct=key_arg[const.keep_pct],
                         max_iters=20, curve=curve)
        solver_list.append(solver)
    # best_mimic, best_score, best_time, best_idx, best_curve\
    #     = solver_pick(problem, solver_list, iteration, seed, criteria, maximum)
    best_mimic, best_score, best_time, best_idx, best_curve\
        = solver_pick(problem, solver_list, iteration, seed, criteria, maximum)
    key_arg_list = np.array(key_arg_list)
    return best_mimic, best_score, best_time, key_arg_list[best_idx], best_curve


def pick_shorttest_time(list_to_pick, times):
    best_solver_idx = np.argwhere(times == min(times))
    list_to_pick = np.array(list_to_pick)

    return list_to_pick[best_solver_idx]


def experiment(problem, solvers, iteration, seed, criteria=0, maximum=True, mimic_iteration=None):
    solver_name = None
    mimic_iteration = iteration if mimic_iteration is None else mimic_iteration
    if solvers is None:
        print(f"{problem}_rhc")
        best_rhc, bs_rhc, bf_rhc, bk_rhc, bc_rhc = rhc_pick(problem, iteration, seed, criteria=0, maximum=True)
        print(f"{problem}_sa")
        best_sa, bs_sa, bf_sa, bk_sa, bc_sa = sa_pick(problem, iteration, seed, criteria, maximum)
        print(f"{problem}_ga")
        best_ga, bs_ga, bf_ga, bk_ga, bc_ga = ga_pick(problem, iteration, seed, criteria, maximum)
        print(f"{problem}_mimic")
        best_mimic, bs_mimic, bf_mimic, bk_mimic, bc_mimic = \
            mimic_pick(problem, mimic_iteration, seed, criteria, maximum)
        score_collection = [bs_rhc, bs_sa, bs_ga, bs_mimic]
        time_collection = [bf_rhc, bf_sa, bf_ga, bf_mimic]
        curve_collection = [bc_rhc, bc_sa, bc_ga, bc_mimic]

        # score_collection = [bs_rhc, bs_sa]
        # time_collection = [bf_rhc, bf_sa]
        # curve_collection = [bc_rhc, bc_sa]

        solver_scores = [pick_shorttest_time(score_collection[i],
                                             time_collection[i]) for i in range(len(time_collection))]
        # solver_times = [pick_shorttest_time(time_collection[i],
        #                                    time_collection[i]) for i in range(len(time_collection))]
        solver_times = [min(time_collection[i]) for i in range(len(time_collection))]
        solver_curve = [pick_shorttest_time(curve_collection[i],
                                            time_collection[i]) for i in range(len(time_collection))]
        solver_name = [const.rhc, const.sa, const.ga, const.mimic]

        # rhc_score = pick_shorttest_time(bs_rhc, bf_rhc)
        # rhc_curve = pick_shorttest_time(bc_rhc, bf_rhc)
        # sa_score = pick_shorttest_time(bs_sa, bf_sa)
        # sa_curve =

    else:
        solver_scores = []
        solver_times = []
        solver_curve = []
        for solver in solvers:
            state_collection, fitness_collection, curve_collection, time_taken = \
                solver_simulation(problem, solver, iteration, seed)
            solver_scores.append(np.mean(fitness_collection))
            solver_times.append(np.mean(time_taken))
            solver_curve.append(curve_collection[0])

    return solver_scores, solver_times, solver_curve, solver_name


def curve_graphing_with_format(curves, legends, title):
    curves = [cur[0][0] for cur in curves]
    curves = np.array(curves)
    graphing(curves, legends, title)


def graphing(curves, legends, title):
    for line in curves:
        plt.plot(np.arange(1, len(line)+1), line)
    plt.legend(legends)
    plt.title(title)
    plt.savefig(title)
    plt.close()


def recorder(text: str, file_name):
    pd.to_pickle(text, f"{file_name}.pkl")


def experiment_graphing(problem, solvers, iteration, seed, title, criteria=0, maximum=True, mimic_iteration=1):
    solver_scores, solver_times, solver_curve, solver_name \
        = experiment(problem, solvers, iteration, seed, criteria, maximum, mimic_iteration)
    curve_graphing_with_format(solver_curve, solver_name, title)
    for i in range(len(solver_name)):
        string = f"{solver_name[i]}: best score: {solver_scores[i]}, avg time: {solver_times[i]}"
        print(string)
        recorder(string, f"{problem}_{solver_name[i]}_best_pick")


# def approximate_time_experiment(problem, iteration, seed, time_limit):
#     """
#     iteration: simulation times to calculate avg training time
#     """
#     # problem = problem_generation()
#     rhc_solver_1_iter = partial(mlrose_hiive.random_hill_climb, max_iters=1)
#     sa_solverr_1_iter = partial(mlrose_hiive.simulated_annealing, max_iters=1)
#     ga_solverr_1_iter = partial(mlrose_hiive.simulated_annealing, max_iters=1)
#     mimic_solverr_1_iter = partial(mlrose_hiive.mimic, max_iters=1)
#     solvers_1_iter = [rhc_solver_1_iter, sa_solverr_1_iter, ga_solverr_1_iter, mimic_solverr_1_iter]
#     one_iter_time = []
#     for sol in solvers_1_iter:
#         state_collection, fitness_collection, curve_collection, time_taken = \
#             solver_simulation(problem, sol, iteration, seed)
#         avg_time_taken = np.mean(time_taken)
#         one_iter_time.append(avg_time_taken)
#     max_iters = []
#     for times in one_iter_time:
#         max_iters.append(int(np.floor(time_limit/times)))
#     rhc_solver = partial(mlrose_hiive.random_hill_climb, max_iters=max_iters[0])
#     sa_solver = partial(mlrose_hiive.simulated_annealing, max_iters=max_iters[1])
#     ga_solver = partial(mlrose_hiive.simulated_annealing, max_iters=max_iters[2])
#     mimic_solver = partial(mlrose_hiive.mimic, max_iters=max_iters[3])
#     solvers = [rhc_solver, sa_solver, ga_solver, mimic_solver]
#     solver_fitness = []
#     solver_time_taken = []
#     solver_curve = []
#     for sol in solvers:
#         t1 = time.time()
#         state_collection, fitness_collection, curve_collection, time_taken = \
#             solver_simulation(problem, sol, iteration, seed)
#         print(f"{sol}, time: {time.time()-t1}")
#         solver_fitness.append(np.mean(fitness_collection))
#         solver_time_taken.append(np.mean(time_taken))
#         solver_curve.append(curve_collection[0])
#     solver_names = [const.rhc, const.sa, const.ga, const.mimic]
#     curve_graphing_with_format(solver_curve, solver_names, "approx_time_curve")
#     for i in range(4):
#         string = f"{solver_names[i]}, score: {solver_fitness[i]}, time_taken: {solver_time_taken[i]}"
#         print(string)
#         recorder(string, f"{solver_names[i]}_approx_time")


# if __name__ == '__main__':
    # problem4 = mlrose_hiive.DiscreteOpt(length=500, fitness_fn=mlrose_hiive.FourPeaks(),
    #                                     maximize=True, max_val=2)
    # import config
    # problem2 = mlrose_hiive.FlipFlopOpt(length=500)
    # approximate_time_experiment(problem2, 10, config.seed,)
#     # problem3 = mlrose_hiive.Ko
#
#     experiment_graphing(problem2, None, 5, None, 'AlternativeOneExp_BestAlgoCurve.png')
#     experiment_graphing(problem4, None, 5, None, 'FourPeaksExp_BestAlgoCurve.png')
