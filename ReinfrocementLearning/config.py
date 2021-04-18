import numpy as np
import hiive.mdptoolbox.example as example

discount_rate_collection = np.arange(0.6, 1, 0.05)
rand_seed = 311231

ql_lr_collection = [0.05, 0.1, 0.2]
ql_epsilon_decay_collection = [0.9, 0.95, 0.99, 0.999]
# ql_training_time =

sf_p, sf_r = example.forest(S=10, r1=2, r2=4)
bf_p, bf_r = example.forest(S=100, r1=100, r2=200)

small_forest_reward_num_step = 20
small_forest_reward_num_trail = 10000

big_forest_reward_num_step = 250
big_forest_reward_num_trail = 400
