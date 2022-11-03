from random import random

import numpy as np
from matplotlib import pyplot as plt

from utilis4 import *


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji


def wahadlo_uczenie(episode_count=1_000, ):
    W = np.random.rand(FEATURE_COUNT, RESOLUTION)

    max_steps = 1000
    MSE_ALL = []
    MSE_PLOT = []
    SCORE_PLOT = []
    lr = 1e-8

    for episode in range(episode_count):
        state = BEGIN_STATES[0]
        batch = []

        # Learning loop
        for i in range(max_steps):
            s = encode_states(state)
            action = predict_action(s, W)
            state_next = wahadlo(state, action)
            r = reward(state_next)
            batch.append([s, action, r])

            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                break

            state = state_next
        return

        n = len(batch)
        S = [x[0] for x in batch]
        A = [x[1] for x in batch]
        R = [x[2] for x in batch]
        S_A = np.array([x[0] + [x[1]] for x in batch])
        MSE_batches = []
        for i in range(n):
            Y_hat = np.array([predict_reward(s, a, W) for s, a in zip(S, A)])
            MSE = (1 / n) * np.sum(R - Y_hat)
            for wi in range(FEATURE_COUNT):
                derivative_wi = (-2 / n) * np.sum(S_A[:, wi] * (R - Y_hat))
                W[wi] -= lr * derivative_wi
                MSE_batches.append(MSE)
        MSE_ALL.append(sum(MSE_batches) / len(MSE_batches))
        # print(MSE)

        if episode % 100 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, W)
            # progress = round((episode / episode_count) * 100, 2)
            MSE_PLOT.append(MSE_ALL[-1])
            SCORE_PLOT.append(score)
            print(f"episode: {episode} - score: {score}  steps: {steps}  MSE={MSE_ALL[-1]}  lr:{lr}")
            if episode >= 100:
                lr = 1e-9
            if episode >= 300:
                lr = 1e-10
            if episode >= 600:
                lr = 1e-12
            if episode >= 800:
                lr = 1e-13
            # print(f"{progress}% MSE={MSE_ALL[-1]}")
    plt.plot(MSE_PLOT)
    plt.plot(SCORE_PLOT)
    plt.title(f"MSE lr:{lr} epochs:{episode_count}")
    plt.xlabel("Kroki")
    plt.ylabel("MSE")
    plt.show()
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
