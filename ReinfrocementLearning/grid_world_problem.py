import numpy as np


def grid_wrold_problem(length_side=5, penalty=-10, reward=10, step_penalty=-0.01, p=0.75, restart=True):
    """
    0: up
    1: right
    2: down
    3: left

    the trap is in (num_row-1, num_col)
    """
    num_state = length_side * length_side
    num_state_with_prison = num_state + 1
    num_actions = 4
    P = np.zeros((num_actions, num_state_with_prison, num_state_with_prison))
    # for i in range(num_actions):
    for j in range(num_state):
        cur_row = j // length_side
        cur_col = j % length_side

        upper_state = max(cur_row-1, 0) * length_side + cur_col
        right_state = cur_row * length_side + min(cur_col+1, length_side-1)
        down_state = min(cur_row+1, length_side-1) * length_side + cur_col
        left_state = cur_row*length_side + max(cur_col-1, 0)

        if (cur_row == length_side-1 and cur_col == length_side-1) \
                or (cur_row == length_side-2 and cur_col == length_side-1):
            P[0][j][num_state] = 1
            P[1][j][num_state] = 1
            P[2][j][num_state] = 1
            P[3][j][num_state] = 1
        else:
            P[0][j][upper_state] += p
            P[0][j][right_state] += (1-p)

            P[1][j][right_state] += p
            P[1][j][down_state] += (1-p)

            P[2][j][down_state] += p
            P[2][j][left_state] += (1-p)

            P[3][j][left_state] += p
            P[3][j][upper_state] += (1-p)
    #
    for i in range(num_actions):
        if not restart:
            P[i][num_state][num_state] = 1
        else:
            P[i][num_state][0] = 1

    R = np.zeros((num_state_with_prison, num_actions))
    for i in range(num_state):
        for j in range(num_actions):
            cur_row = i // length_side
            cur_col = i % length_side
            if cur_row == length_side - 1 and cur_col == length_side - 1:
                R[i][j] = reward
            elif cur_row == length_side - 2 and cur_col == length_side - 1:
                R[i][j] = penalty
            else:
                R[i][j] = step_penalty

    return (P, R)

# if __name__ == '__main__':
#     p, r = grid_wrold_problem(5, -3, 3, -0.1, 0.75)
