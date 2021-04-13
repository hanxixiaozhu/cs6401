from grid_world_problem import grid_wrold_problem as gp
# import hiive.mdptoolbox.mdp as mdp
import mdp_revised as mdp
import numpy as np
import const
import utils
import pandas as pd
import config
import matplotlib.pyplot as plt


def value_iter_compare(sgw_p, sgw_r, initial_value_collection, discount_rate_collection, reward_num_step,
                       reward_num_trial):
    """
    problem with default setting

    Test for different initial values & different discount factor
    """
    parameter_list = []

    vi_collections = []
    for ivs in initial_value_collection:
        for dr in discount_rate_collection:
            parameter_list.append({const.initial_values: ivs, const.discount_rate: dr})
            vi = mdp.ValueIteration(sgw_p, sgw_r, initial_value=ivs, gamma=dr)
            vi_collections.append(vi)

    msg_collections = []
    reward_collection = []
    path_collection = []
    for vi in vi_collections:
        msg = vi.run()
        msg_collections.append(msg)
        # rewards, paths = utils.policy_reward_cal(25, vi.policy, sgw_p, sgw_r, vi.gamma, 1000)
        rewards, paths = utils.policy_reward_cal(reward_num_step, vi.policy, sgw_p, sgw_r, vi.gamma, reward_num_trial)
        reward_collection.append(rewards)
        path_collection.append(paths)
    return parameter_list, vi_collections, msg_collections, reward_collection, path_collection


def small_vi_exp():
    sgw_p, sgw_r = gp()

    num_state = len(sgw_p[0])
    initial_value_collection = [np.zeros(num_state), np.random.uniform(size=num_state), np.zeros(num_state)+50]
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = 10
    reward_num_trial = 10000
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = \
        value_iter_compare(sgw_p, sgw_r, initial_value_collection, discount_rate_collection,
                           reward_num_step, reward_num_trial)
    return parameter_list, vi_collections, msg_collections, reward_collection, path_collection


def big_vi_exp():
    sgw_p, sgw_r = gp(length_side=50, penalty=-1000, reward=1000)

    num_state = len(sgw_p[0])
    initial_value_collection = [np.zeros(num_state), np.random.uniform(size=num_state), np.zeros(num_state)+50]
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = 200
    reward_num_trial = 2000
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = \
        value_iter_compare(sgw_p, sgw_r, initial_value_collection, discount_rate_collection,
                           reward_num_step, reward_num_trial)
    return parameter_list, vi_collections, msg_collections, reward_collection, path_collection


def vi_label_creation():
    vi_discount_collection = config.discount_rate_collection
    initial_value_collection = const.vi_initial_values_set_name
    param_label_collection = []
    for ivs in initial_value_collection:
        for dr in vi_discount_collection:
            label = ivs + '_' + str(np.round(dr, 2))
            param_label_collection.append(label)
    return param_label_collection


def vi_one_dimension_stat_compare(one_d_collection):
    rmc = np.array(one_d_collection)
    rmc = rmc.reshape((3, 8))
    col = np.round(config.discount_rate_collection, 2)
    row = const.vi_initial_values_set_name
    table = pd.DataFrame(rmc, columns=col, index=row)
    return table


def vi_error_evolve_plot(title, error_evolve_collection):
    param_legend = vi_label_creation()
    for i in range(3):
        sub_title = 'vi_' + title + '_' + const.vi_initial_values_set_name[i]
        for j in range(8):
            ee = error_evolve_collection[i*8 + j]
            x_points = np.arange(len(ee))
            plt.plot(x_points, ee)
        plt.title(sub_title)
        start_idx = i*8
        plt.legend(param_legend[start_idx: start_idx+8])
        plt.savefig(sub_title)
        plt.close()
    # for ee in error_evolve_collection:
    #     x_points = np.arange(len(ee))
    #     # print(x_points)
    #     plt.plot(x_points, ee)
    # plt.title(title)
    # param_legend = value_label_creation()
    # plt.legend(param_legend)
    # plt.savefig(title)
    # plt.close()


def vi_exp_analysis(msg_collections, reward_collection, title):
    time_taken_collections, iteration_taken_collection, error_evolve_collection, \
        reward_mean_collection, reward_std_collection = utils.result_analysis(msg_collections, reward_collection)
    reward_mean_table = vi_one_dimension_stat_compare(reward_mean_collection)
    reward_std_table = vi_one_dimension_stat_compare(reward_std_collection)
    time_taken_table = vi_one_dimension_stat_compare(time_taken_collections)
    iteration_taken_table = vi_one_dimension_stat_compare(iteration_taken_collection)
    vi_error_evolve_plot(title, error_evolve_collection)
    # graph_pd_table(reward_mean_table, title)
    print("Mean reward table")
    print(reward_mean_table.to_string())
    print("Reward std table")
    print(reward_std_table.to_string())
    print("Time Taken table")
    print(time_taken_table.to_string())
    print("Iteration taken table")
    print(iteration_taken_table.to_string())
    return reward_mean_table, reward_std_table, time_taken_table, iteration_taken_table


def small_vi_exp_analysis():
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = small_vi_exp()
    print("Small grid world Value Iteration Exp")
    srt, sst, stt, sit = vi_exp_analysis(msg_collections, reward_collection, const.grid_world_small)
    return srt, sst, stt, sit


def big_vi_exp_analysis():
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = big_vi_exp()
    print("Big grid world Value Iteration Exp")
    brt, bst, btt, bit = vi_exp_analysis(msg_collections, reward_collection, const.grid_world_big)
    return brt, bst, btt, bit


def vi_exp():
    small_vi_exp_analysis()
    big_vi_exp_analysis()


def policy_iter_compare(sgw_p, sgw_r,  initial_policy_collection, discount_rate_collection, reward_num_step,
                        reward_num_trial):
    parameter_list = []

    pi_collections = []
    for ip in initial_policy_collection:
        for dr in discount_rate_collection:
            parameter_list.append({const.initial_policy: ip, const.discount_rate: dr})
            pi = mdp.PolicyIteration(sgw_p, sgw_r, policy0=ip, gamma=dr)
            pi_collections.append(pi)

    msg_collections = []
    reward_collection = []
    path_collection = []
    for pi in pi_collections:
        msg = pi.run()
        msg_collections.append(msg)
        # rewards, paths = utils.policy_reward_cal(25, vi.policy, sgw_p, sgw_r, vi.gamma, 1000)
        rewards, paths = utils.policy_reward_cal(reward_num_step, pi.policy, sgw_p, sgw_r, pi.gamma, reward_num_trial)
        reward_collection.append(rewards)
        path_collection.append(paths)
    return parameter_list, pi_collections, msg_collections, reward_collection, path_collection


def small_pi_exp():
    sgw_p, sgw_r = gp()

    num_state = len(sgw_p[0])
    initial_policy_collection = [np.zeros(num_state), np.random.choice([0, 1, 2, 3], size=num_state)]
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = 10
    reward_num_trial = 10000
    parameter_list, pi_collections, msg_collections, reward_collection, path_collection = \
        policy_iter_compare(sgw_p, sgw_r, initial_policy_collection, discount_rate_collection,
                            reward_num_step, reward_num_trial)
    return parameter_list, pi_collections, msg_collections, reward_collection, path_collection


def big_pi_exp():
    sgw_p, sgw_r = gp(length_side=50, penalty=-1000, reward=1000)

    num_state = len(sgw_p[0])
    initial_policy_collection = [np.zeros(num_state), np.random.choice([0, 1, 2, 3], size=num_state)]
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = 200
    reward_num_trial = 2000
    parameter_list, pi_collections, msg_collections, reward_collection, path_collection = \
        policy_iter_compare(sgw_p, sgw_r, initial_policy_collection, discount_rate_collection,
                            reward_num_step, reward_num_trial)
    return parameter_list, pi_collections, msg_collections, reward_collection, path_collection


def pi_label_creation():
    vi_discount_collection = config.discount_rate_collection
    initial_policy_collection = const.pi_initial_policies_set_name
    param_label_collection = []
    for ip in initial_policy_collection:
        for dr in vi_discount_collection:
            label = ip + '_' + str(np.round(dr, 2))
            param_label_collection.append(label)
    return param_label_collection


def pi_one_dimension_stat_compare(one_d_collection):
    rmc = np.array(one_d_collection)
    rmc = rmc.reshape((2, 8))
    col = np.round(config.discount_rate_collection, 2)
    row = const.pi_initial_policies_set_name
    table = pd.DataFrame(rmc, columns=col, index=row)
    return table


def pi_error_evolve_plot(title, error_evolve_collection):
    param_legend = pi_label_creation()
    for i in range(2):
        sub_title = 'pi' + title + '_' + const.pi_initial_policies_set_name[i]
        for j in range(8):
            ee = error_evolve_collection[i*8 + j]
            x_points = np.arange(len(ee))
            plt.plot(x_points, ee)
        plt.title(sub_title)
        start_idx = i*8
        plt.legend(param_legend[start_idx: start_idx+8])
        plt.savefig(sub_title)
        plt.close()


def pi_exp_analysis(msg_collections, reward_collection, title):
    time_taken_collections, iteration_taken_collection, error_evolve_collection, \
        reward_mean_collection, reward_std_collection = utils.result_analysis(msg_collections, reward_collection)
    reward_mean_table = pi_one_dimension_stat_compare(reward_mean_collection)
    reward_std_table = pi_one_dimension_stat_compare(reward_std_collection)
    time_taken_table = pi_one_dimension_stat_compare(time_taken_collections)
    iteration_taken_table = pi_one_dimension_stat_compare(iteration_taken_collection)
    pi_error_evolve_plot(title, error_evolve_collection)
    # graph_pd_table(reward_mean_table, title)
    print("Mean reward table")
    print(reward_mean_table.to_string())
    print("Reward std table")
    print(reward_std_table.to_string())
    print("Time Taken table")
    print(time_taken_table.to_string())
    print("Iteration taken table")
    print(iteration_taken_table.to_string())
    return reward_mean_table, reward_std_table, time_taken_table, iteration_taken_table


def small_pi_exp_analysis():
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = small_pi_exp()
    print("Small grid world Policy Iteration Exp")
    srt, sst, stt, sit = pi_exp_analysis(msg_collections, reward_collection, const.grid_world_small)
    return srt, sst, stt, sit


def big_pi_exp_analysis():
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = big_pi_exp()
    print("Big grid world Policy Iteration Exp")
    brt, bst, btt, bit = pi_exp_analysis(msg_collections, reward_collection, const.grid_world_big)
    return brt, bst, btt, bit


def pi_exp():
    small_pi_exp_analysis()
    big_pi_exp_analysis()


def q_learning_compare(sgw_p, sgw_r,  lr_collection, epsilon_decay_collection, discount_rate_collection,
                       reward_num_step, reward_num_trial, ql_iteration=10000):
    parameter_list = []

    ql_collections = []
    for lr in lr_collection:
        for eps in epsilon_decay_collection:
            for dr in discount_rate_collection:
                parameter_list.append({const.learning_rate: lr, const.epsilon_decay: eps, const.discount_rate: dr})
                ql = mdp.QLearning(sgw_p, sgw_r, alpha=lr, epsilon=eps, gamma=dr, n_iter=ql_iteration)
                ql_collections.append(ql)

    msg_collections = []
    reward_collection = []
    path_collection = []
    for ql in ql_collections:
        msg = ql.run()
        msg_collections.append(msg)
        # rewards, paths = utils.policy_reward_cal(25, vi.policy, sgw_p, sgw_r, vi.gamma, 1000)
        rewards, paths = utils.policy_reward_cal(reward_num_step, ql.policy, sgw_p, sgw_r, ql.gamma, reward_num_trial)
        reward_collection.append(rewards)
        path_collection.append(paths)
    return parameter_list, ql_collections, msg_collections, reward_collection, path_collection


def small_ql_exp():
    sgw_p, sgw_r = gp()

    lr_collection = config.ql_lr_collection
    epsilon_decay_collection = config.ql_epsilon_decay_collection
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = 10
    reward_num_trial = 10000
    parameter_list, ql_collections, msg_collections, reward_collection, path_collection = \
        q_learning_compare(sgw_p, sgw_r, lr_collection, epsilon_decay_collection, discount_rate_collection,
                           reward_num_step, reward_num_trial)
    return parameter_list, ql_collections, msg_collections, reward_collection, path_collection


def big_ql_exp():
    sgw_p, sgw_r = gp(length_side=50, penalty=-1000, reward=1000)

    lr_collection = config.ql_lr_collection
    epsilon_decay_collection = config.ql_epsilon_decay_collection
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = 200
    reward_num_trial = 2000
    parameter_list, ql_collections, msg_collections, reward_collection, path_collection = \
        q_learning_compare(sgw_p, sgw_r, lr_collection, epsilon_decay_collection, discount_rate_collection,
                           reward_num_step, reward_num_trial)
    return parameter_list, ql_collections, msg_collections, reward_collection, path_collection


def ql_label_creation():
    lr_collection = config.ql_lr_collection
    epsilon_collection = config.ql_epsilon_decay_collection
    dr_collection = config.discount_rate_collection
    param_label_collection = []
    for lr in lr_collection:
        for eps in epsilon_collection:
            for dr in dr_collection:
                label = str(np.round(lr, 2)) + '_' + str(np.round(eps, 2)) + '_' + str(np.round(dr, 2))
                param_label_collection.append(label)
    return param_label_collection


def ql_one_dimension_stat_compare(one_d_collection):
    rmc = np.array(one_d_collection)
    rmc = rmc.reshape((3, 4, 8))
    col = np.round(config.discount_rate_collection, 2)
    row = config.ql_epsilon_decay_collection
    keys = config.ql_lr_collection
    table_dicts = {}
    for i, k in enumerate(keys):
        table_dicts[k] = pd.DataFrame(rmc[i], columns=col, index=row)
    # table = pd.DataFrame(rmc, columns=col, index=row)
    return table_dicts


def ql_error_evolve_plot(title, error_evolve_collection):
    param_legend = ql_label_creation()
    for i in range(3):
        for j in range(4):
            sub_title = 'ql' + title + '_lr_' + str(np.around(config.ql_lr_collection[i], 2)) + '_eps_' + \
                        str(np.around(config.ql_epsilon_decay_collection[j], 2)) + '.png'
            for k in range(8):
                ee = error_evolve_collection[i*32 + j*8 + k]
                x_points = np.arange(len(ee))
                plt.plot(x_points, ee)
            plt.title(sub_title)
            start_idx = j*8 + i*32
            plt.legend(param_legend[start_idx: start_idx+8])
            plt.savefig(sub_title)
            plt.close()


def ql_exp_analysis(msg_collections, reward_collection, title):
    time_taken_collections, iteration_taken_collection, error_evolve_collection, \
        reward_mean_collection, reward_std_collection = utils.result_analysis(msg_collections, reward_collection)
    reward_mean_table_dict = ql_one_dimension_stat_compare(reward_mean_collection)
    reward_std_table_dict = ql_one_dimension_stat_compare(reward_std_collection)
    time_taken_table_dict = ql_one_dimension_stat_compare(time_taken_collections)
    iteration_taken_table_dict = ql_one_dimension_stat_compare(iteration_taken_collection)
    ql_error_evolve_plot(title, error_evolve_collection)
    # graph_pd_table(reward_mean_table, title)
    lr_collection = config.ql_lr_collection
    for lr in lr_collection:
        print(f"Mean reward table with learning rate: {lr}")
        print(reward_mean_table_dict[lr].to_string())
        print(f"Reward std table with learning rate: {lr}")
        print(reward_std_table_dict[lr].to_string())
        print(f"Time Taken table with learning rate: {lr}")
        print(time_taken_table_dict[lr].to_string())
        print(f"Iteration taken table with learning rate: {lr}")
        print(iteration_taken_table_dict[lr].to_string())
    return reward_mean_table_dict, reward_std_table_dict, time_taken_table_dict, iteration_taken_table_dict


def small_ql_exp_analysis():
    parameter_list, ql_collections, msg_collections, reward_collection, path_collection = small_ql_exp()
    print("Small grid world Q Learning Exp")
    srt, sst, stt, sit = ql_exp_analysis(msg_collections, reward_collection, const.grid_world_small)
    return srt, sst, stt, sit


def big_ql_exp_analysis():
    parameter_list, ql_collections, msg_collections, reward_collection, path_collection = big_ql_exp()
    print("Big grid world Q Learning Exp")
    brt, bst, btt, bit = ql_exp_analysis(msg_collections, reward_collection, const.grid_world_big)
    return brt, bst, btt, bit


def ql_exp():
    small_ql_exp_analysis()
    big_ql_exp_analysis()


if __name__ == '__main__':
    np.random.seed(config.rand_seed)
    # vi_exp()
    # pi_exp()
    ql_exp()
