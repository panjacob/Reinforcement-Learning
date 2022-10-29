import numpy as np

from utilis import *


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji

def wahadlo_uczenie(alpha=0.2, epsilon=0.1, episode_count=10_000, warmup_steps=20_000):
    Q = np.zeros([RESOLUTION[0], RESOLUTION[1], RESOLUTION[2], RESOLUTION[3], RESOLUTION[4]], dtype=float)
    # print(Q.size)
    # print(Q.itemsize)
    # print(Q.size * Q.itemsize / 1_000_000)
    Q_size = Q.size
    is_warmup = True
    max_steps = 100
    for episode in range(episode_count):
        if is_warmup and episode < warmup_steps:
            epsilon = 1
        elif is_warmup and episode > warmup_steps:
            is_warmup = False
            epsilon = 0.1
            print('end warmup <---------------------------------------')

        # state_start = episode % BEGIN_STATES_COUNT
        # state = BEGIN_STATES[state_start, :]
        state = BEGIN_STATES[0]
        W = np.zeros(max_steps, dtype=float)
        HISTORY = []

        for i in range(max_steps):
            F, F_index = random_action() if random() < epsilon else best_action(state, Q)
            s = encode_states(state)
            state_new = wahadlo(state, F)

            # if abs(state_new[0]) > BIN_MAX[0] or abs(state_new[2]) > BIN_MAX[2]:

            R = reward(state, state, F)
            Rn = reward(state, state_new, F)
            # alpha = wielkość kroku
            # w = w - alpha * error
            # Q[(s[0], s[1], s[2], s[3], F_index)] = Rn ** 2
            # W[i] = (alpha ** i) * R
            # error = (Q[(s[0], s[1], s[2], s[3], F_index)]  - Rn) ** 2
            l_gradient = Rn - R
            # W[i] = Q[(s[0], s[1], s[2], s[3], F_index)] - (alpha ** i) * error
            W[i] = (0.3 ** i) * l_gradient ** 2
            # W[i] = Rn ** 2
            HISTORY.append((s[0], s[1], s[2], s[3], F_index))
            # Q[state_F] = Q[state_F] + alpha * (R + gamma * reward_next - Q[state_F])
            # Q[state_F] = Q[state_F] + alpha * (R + gamma * reward_next)
            state = state_new
            if abs(state_new[0]) >= np.pi / 2 or abs(state_new[2]) > BIN_MAX[2]:
                break

        for i, state in enumerate(HISTORY):
            # print(np.sum(W[i:]))
            Q[state] = np.sum(W[i:])

        if episode % 1000 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, Q)
            progress = round((episode / episode_count) * 100, 2)
            # visited = round(np.count_nonzero(Q) / Q_size * 100, 6)
            visited = np.count_nonzero(Q)
            print(f"{progress}% - score: {score}  steps: {steps}   visited: {visited}")
            print(np.min(Q[np.nonzero(Q)]), np.max(Q[np.nonzero(Q)]))

    return wahadlo_test(BEGIN_STATES, Q)


# best_steps = 0
# best_steps_params = ()
# best_score = -9999
# best_score_params = ()
# for alpha in np.arange(0.1, 1, 0.2):
#     for gamma in np.arange(0.1, 1, 0.2):
#         for epsilon in np.arange(0.1, 1, 0.2):
#             score, steps = wahadlo_uczenie(alpha, gamma, epsilon)
#             if steps > best_steps:
#                 best_steps = steps
#                 print('best steps: ', steps, (alpha, gamma, epsilon))
#             if score > best_score:
#                 best_score = score
#                 print('best score: ', score, (alpha, gamma, epsilon))

wahadlo_uczenie(0.9, 0.1, 1_000_000)
