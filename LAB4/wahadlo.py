import numpy as np

from utilis import *


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji

def wahadlo_uczenie(alpha=0.5, gamma=0.5, epsilon=0.1, episode_count=10_000, warmup_steps=20_000):
    Q = np.random.rand([RESOLUTION[0], RESOLUTION[1], RESOLUTION[2], RESOLUTION[3], RESOLUTION[4]], dtype=float)
    is_warmup = True
    max_steps = 100
    for episode in range(episode_count):
        # if is_warmup and episode < warmup_steps:
        #     epsilon = 1
        # elif is_warmup and episode > warmup_steps:
        #     is_warmup = False
        #     epsilon = 0.1
        #     print('end warmup <---------------------------------------')

        # state_start = episode % BEGIN_STATES_COUNT
        # state = BEGIN_STATES[state_start, :]
        state = BEGIN_STATES[0]
        D = []

        for i in range(max_steps):
            F, F_index = random_action() if random() < epsilon else calc_best_action(state, Q)
            s = encode_states(state)

            state_next = wahadlo(state, F)
            sn = encode_states(state_next)

            R = reward(state, state, F)

            _, F_index_next = calc_best_action(state_next, Q)
            Q_max_next = Q[(sn[0], sn[1], sn[2], sn[3], F_index_next)]
            Q_max = Q[(s[0], s[1], s[2], s[3], F_index)]
            Q_gradient = gradient(Q[s[0], s[1], s[2], s[3], :])
            W_delta = alpha * ((R + gamma + Q_max_next) - Q_max) * Q_gradient

            Q[s[0], s[1], s[2], s[3], :] += W_delta
            print(Q[s[0], s[1], s[2], s[3], :])

            D.append((s[0], s[1], s[2], s[3], F_index))
            state = state_next
            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                break

        # for i, state in enumerate(HISTORY):
        #     # print(np.sum(W[i:]))
        #     Q[state] = np.sum(W[i:])

        if episode % 1000 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, Q)
            progress = round((episode / episode_count) * 100, 2)
            # visited = round(np.count_nonzero(Q) / Q_size * 100, 6)
            visited = np.count_nonzero(Q)
            print(f"{progress}% - score: {score}  steps: {steps}   visited: {visited}")
            # print(np.min(Q[np.nonzero(Q)]), np.max(Q[np.nonzero(Q)]))

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

wahadlo_uczenie()
