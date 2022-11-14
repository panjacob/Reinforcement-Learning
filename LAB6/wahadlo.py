# from random import random
import copy
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt

from utilis import *


def wahadlo_uczenie(episode_count=50_000, alpha=0.0001, gamma=0.998, epsilon=0.5):
    W = np.random.rand(FEATURE_COUNT, RESOLUTION, RESOLUTION_A)
    max_steps = 1000
    best_score = -9999999
    best_W = None
    best_steps = 0

    for episode in range(episode_count):
        state = BEGIN_STATES[episode % BEGIN_STATES_COUNT]
        history = []
        for i in range(max_steps):
            r = reward(state)
            s = encode_states(state)
            a, r_hat = random_action(s, W) if random.random() < epsilon else predict_action(s, W)
            action = BINS[4][a]

            state_next = wahadlo(state, action)
            s_next = encode_states(state_next)
            a_next, r_next_hat = predict_action(s_next, W)
            delta = r + gamma * r_next_hat - r_hat
            history.append((s, a, delta))

            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                break

            state = state_next

        # update W
        for i, (s, a, _) in enumerate(history):
            delta_sum = 0
            for d_i in range(0, i + 1):
                delta_sum += history[d_i][2]

            gradient = one_hot_encoding_state(s)
            W[:, :, a] += alpha * delta_sum * gradient
            W[:, :, a] = W[:, :, a] / np.linalg.norm(W[:, :, a])

        if episode % 100 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, W)

            if steps >= best_steps:
                best_score = score
                best_steps = steps
                best_W = copy.deepcopy(W)
                prevented = False
            else:
                prevented = True
                W = copy.deepcopy(best_W)

            print(f"episode: {episode} - score: {score}  steps: {steps}  best_score:{best_score} "
                  f" best_steps:{best_steps}  {'Uzywam starych wag' if prevented else ''}")
            if steps >= 1000:
                break

    score, steps = wahadlo_test(BEGIN_STATES, best_W)
    print(f"Ostatecznie: score: {score}  steps: {steps}")
    return 0


wahadlo_uczenie()
