import numpy as np
from utilis import encode_states, RESOLUTION, BINS, decode_state

Q = np.random.rand(RESOLUTION[0], RESOLUTION[1], RESOLUTION[2], RESOLUTION[3], RESOLUTION[4])

# def best_action(state_encoded, Q):
#     action_index = np.argmax(Q[state_encoded[0], state_encoded[1], state_encoded[2], state_encoded[3], :])
#     action = BINS[4][action_index]
#     return action, action_index
#
#
# def action_score(state_encoded, action_index, Q):
#     return Q[state_encoded[0], state_encoded[1], state_encoded[2], state_encoded[3], action_index]
#
#
# action_index = 20
# new_score = 999
# state = [0, 0, 0, 0]
#
# s = encode_states(state)
# print(s)
# Q[s[0], s[1], s[2], s[3], action_index] = new_score
# print(Q[s[0], s[1], s[2], s[3], :])
# value, index = best_action(s, Q)
# score = action_score(s, action_index, Q)
# print(value, score)
