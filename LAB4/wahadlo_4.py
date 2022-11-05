# from random import random

import numpy as np
from matplotlib import pyplot as plt

from utilis4 import *


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji


def wahadlo_uczenie(episode_count=1_000, alpha=0.001, gamma=1, epsilon=0.1):
    W = np.random.rand(FEATURE_COUNT, RESOLUTION)

    max_steps = 1000
    MEs = []

    for episode in range(episode_count):
        state = BEGIN_STATES[0]
        E = []
        for i in range(max_steps):
            r = reward(state)
            s = encode_states(state)
            # random() < epsilon
            a = random_action() if random.random() < epsilon else predict_action(s, W)
            # print(a)

            r_hat = predict_reward(s, a, W)
            state_next = wahadlo(state, a)
            s_next = encode_states(state_next)
            a_next = predict_action(s_next, W)
            r_next_hat = predict_reward(s_next, a_next, W)
            one_hot_states = one_hot_encoding_state(s, a)

            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                break

            W = W + alpha * (r + gamma * r_next_hat - r_hat) * one_hot_states
            # print(np.sum(W))


            state = state_next
        # MEs.append(sum(E) / len(E))

        if episode % 100 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, W)
            print(f"episode: {episode} - score: {score}  steps: {steps}")

    # plt.plot(MEs)
    # plt.xlabel("Kroki")
    # plt.ylabel("ME")
    # plt.show()
    print(W)
    return 0


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
