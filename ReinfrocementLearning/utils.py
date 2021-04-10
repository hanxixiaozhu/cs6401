import numpy as np
import const


def grid_world_policy_visual(policy, length_side):
    po = np.array(policy)
    po = po[:-1]
    po = po.reshape((length_side, length_side))
    maps = {0: '^', 1: '>', 2: 'v', 3: "<"}
    result = []
    for i in range(len(po)):
        tmp = []
        for j in range(len(po[0])):
            tmp.append(maps.get(po[i][j]))
        result.append(tmp)
    result = np.array(result)
    print(result)


def policy_reward_cal(num_step, policy, P, R, gamma, num_iter, initial_state=0):
    """
    num_steo: for each iteration, the number of steps
    """
    reward_collection = []
    path_collection = []
    for iter in range(num_iter):
        reward = 0
        cur_state = initial_state
        path = [cur_state]
        cur_gamma = 1
        for step in range(num_step):
            cur_action = policy[cur_state]
            reward += (R[cur_state][cur_action] * cur_gamma)
            probability = P[cur_action][cur_state]
            cur_state = np.random.choice(list(range(len(probability))), p=probability)
            path.append(cur_state)
            cur_gamma *= gamma
        reward_collection.append(reward)
        path_collection.append(path)
    return reward_collection, path_collection


def msg_analysis(msg_collections):
    """
    analysis msg return by MDP
    @param msg_collections:
    @type msg_collections:
    @return:
    @rtype:
    """
    time_taken_collections = []
    iteration_taken_collection = []
    error_evolve_collection = []
    for i in range(len(msg_collections)):
        msg = msg_collections[i]
        time_taken_collections.append(msg[-1][const.Time])
        iteration_taken_collection.append(msg[-1][const.Iteration])
        error_evolve_collection.append([msg_dict[const.Error] for msg_dict in msg])
    return time_taken_collections, iteration_taken_collection, error_evolve_collection


def reward_collections_analysis(reward_collection):
    """

    @param reward_collection:
    @type reward_collection:
    @return:
    @rtype:
    """
    reward_mean_collection = np.mean(reward_collection, axis=1)
    reward_std_collection = np.std(reward_collection, axis=1)
    return reward_mean_collection, reward_std_collection


def result_analysis(msg_collections, reward_collection):
    """

    @param msg_collections:
    @type msg_collections:
    @param reward_collection:
    @type reward_collection:
    @return:
    @rtype:
    """
    """from msg"""
    time_taken_collections, iteration_taken_collection, error_evolve_collection = msg_analysis(msg_collections)

    """from reward_collection"""
    reward_mean_collection, reward_std_collection = reward_collections_analysis(reward_collection)
    return time_taken_collections, iteration_taken_collection, error_evolve_collection, reward_mean_collection, \
        reward_std_collection
