# import numpy as np
#
#
# def findBestAction(state, Q):
#     best_position = -1
#     max_reward = -999
#     for x in [0, 1, 2, 3]:
#         reward = Q[state[0], state[1], x]
#         if reward > max_reward:
#             max_reward = reward
#             best_position = x
#     return best_position + 1
#
#
# num_of_rows = 3
# num_of_columns = 5
#
# Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)
#
# Q[0, 0, 2] = 7
# xd = max(Q[0, 0, :])
# print(xd)
# print(findBestAction([0, 0], Q))

