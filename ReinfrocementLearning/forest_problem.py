import hiive.mdptoolbox.example as example
import grid_world_exp as gwp
import numpy as np
import config
import const


def small_vi_exp():
    sgw_p, sgw_r = config.sf_p, config.sf_r

    num_state = len(sgw_p[0])
    initial_value_collection = [np.zeros(num_state), np.random.uniform(size=num_state), np.zeros(num_state)+50]
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = config.small_forest_reward_num_step
    reward_num_trial = config.small_forest_reward_num_trail
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = \
        gwp.value_iter_compare(sgw_p, sgw_r, initial_value_collection, discount_rate_collection,
                               reward_num_step, reward_num_trial)
    return parameter_list, vi_collections, msg_collections, reward_collection, path_collection


def big_vi_exp():
    sgw_p, sgw_r = config.bf_p, config.bf_r

    num_state = len(sgw_p[0])
    initial_value_collection = [np.zeros(num_state), np.random.uniform(size=num_state), np.zeros(num_state)+50]
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = config.big_forest_reward_num_step
    reward_num_trial = config.big_forest_reward_num_trail
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = \
        gwp.value_iter_compare(sgw_p, sgw_r, initial_value_collection, discount_rate_collection,
                               reward_num_step, reward_num_trial)
    return parameter_list, vi_collections, msg_collections, reward_collection, path_collection


def small_vi_exp_analysis():
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = small_vi_exp()
    print("Small forest Value Iteration Exp")
    srt, sst, stt, sit = gwp.vi_exp_analysis(msg_collections, reward_collection, const.forest_small)
    return srt, sst, stt, sit


def big_vi_exp_analysis():
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = big_vi_exp()
    print("Big forest Value Iteration Exp")
    brt, bst, btt, bit = gwp.vi_exp_analysis(msg_collections, reward_collection, const.forest_big)
    return brt, bst, btt, bit


def vi_exp():
    small_vi_exp_analysis()
    big_vi_exp_analysis()


def small_pi_exp():
    sgw_p, sgw_r = config.sf_p, config.sf_r

    num_state = len(sgw_p[0])
    initial_policy_collection = [np.zeros(num_state), np.random.choice([0, 1, 2, 3], size=num_state)]
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = config.small_forest_reward_num_step
    reward_num_trial = config.small_forest_reward_num_trail
    parameter_list, pi_collections, msg_collections, reward_collection, path_collection = \
        gwp.policy_iter_compare(sgw_p, sgw_r, initial_policy_collection, discount_rate_collection,
                                reward_num_step, reward_num_trial)
    return parameter_list, pi_collections, msg_collections, reward_collection, path_collection


def big_pi_exp():
    sgw_p, sgw_r = config.bf_p, config.bf_r

    num_state = len(sgw_p[0])
    initial_policy_collection = [np.zeros(num_state), np.random.choice([0, 1, 2, 3], size=num_state)]
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = config.big_forest_reward_num_step
    reward_num_trial = config.big_forest_reward_num_trail
    parameter_list, pi_collections, msg_collections, reward_collection, path_collection = \
        gwp.policy_iter_compare(sgw_p, sgw_r, initial_policy_collection, discount_rate_collection,
                                reward_num_step, reward_num_trial)
    return parameter_list, pi_collections, msg_collections, reward_collection, path_collection


def small_pi_exp_analysis():
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = small_pi_exp()
    print("Small forest Policy Iteration Exp")
    srt, sst, stt, sit = gwp.pi_exp_analysis(msg_collections, reward_collection, const.forest_small)
    return srt, sst, stt, sit


def big_pi_exp_analysis():
    parameter_list, vi_collections, msg_collections, reward_collection, path_collection = big_pi_exp()
    print("Big forest Policy Iteration Exp")
    brt, bst, btt, bit = gwp.pi_exp_analysis(msg_collections, reward_collection, const.forest_big)
    return brt, bst, btt, bit


def pi_exp():
    small_pi_exp_analysis()
    big_pi_exp_analysis()


def small_ql_exp():
    sgw_p, sgw_r = config.sf_p, config.sf_r

    lr_collection = config.ql_lr_collection
    epsilon_decay_collection = config.ql_epsilon_decay_collection
    # discount_rate_collection = np.arange(0.6, 1, 0.05)
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = config.small_forest_reward_num_step
    reward_num_trial = config.small_forest_reward_num_trail
    parameter_list, ql_collections, msg_collections, reward_collection, path_collection = \
        gwp.q_learning_compare(sgw_p, sgw_r, lr_collection, epsilon_decay_collection, discount_rate_collection,
                               reward_num_step, reward_num_trial)
    return parameter_list, ql_collections, msg_collections, reward_collection, path_collection


def big_ql_exp():
    sgw_p, sgw_r = sgw_p, sgw_r = config.bf_p, config.bf_r

    lr_collection = config.ql_lr_collection
    epsilon_decay_collection = config.ql_epsilon_decay_collection
    discount_rate_collection = config.discount_rate_collection
    reward_num_step = config.big_forest_reward_num_step
    reward_num_trial = config.big_forest_reward_num_trail
    parameter_list, ql_collections, msg_collections, reward_collection, path_collection = \
        gwp.q_learning_compare(sgw_p, sgw_r, lr_collection, epsilon_decay_collection, discount_rate_collection,
                               reward_num_step, reward_num_trial)
    return parameter_list, ql_collections, msg_collections, reward_collection, path_collection


def small_ql_exp_analysis():
    parameter_list, ql_collections, msg_collections, reward_collection, path_collection = small_ql_exp()
    print("Small forest Q Learning Exp")
    srt, sst, stt, sit = gwp.ql_exp_analysis(msg_collections, reward_collection, const.forest_small)
    return srt, sst, stt, sit


def big_ql_exp_analysis():
    parameter_list, ql_collections, msg_collections, reward_collection, path_collection = big_ql_exp()
    print("Big forest Q Learning Exp")
    brt, bst, btt, bit = gwp.ql_exp_analysis(msg_collections, reward_collection, const.forest_big)
    return brt, bst, btt, bit


def ql_exp():
    small_ql_exp_analysis()
    big_ql_exp_analysis()


if __name__ == '__main__':
    np.random.seed(config.rand_seed)
    vi_exp()
    pi_exp()
    ql_exp()
    # try:
    #     vi_exp()
    #     print('vi done')
    # except:
    #     print("vi has problem")
    # try:
    #     pi_exp()
    #     print('pi done')
    # except:
    #     print("pi has problem")
    # try:
    #     ql_exp()
    #     print('ql done')
    # except:
    #     print("ql has problem")
