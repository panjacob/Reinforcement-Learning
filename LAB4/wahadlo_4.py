# from random import random

import numpy as np
from matplotlib import pyplot as plt

from utilis4 import *


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji


def wahadlo_uczenie(episode_count=10_000, alpha=0.1, gamma=0.3, epsilon=1):
    W = np.random.rand(FEATURE_COUNT, RESOLUTION)
    max_steps = 1000

    for episode in range(episode_count):
        state = BEGIN_STATES[-1]
        # state = BEGIN_STATES[episode % BEGIN_STATES_COUNT]
        for i in range(max_steps):
            r = reward(state)
            s = encode_states(state)
            a = random_action() if random.random() < epsilon else predict_action(s, W)
            r_hat = predict_reward(s, a, W)
            state_next = wahadlo(state, a)
            s_next = encode_states(state_next)
            a_next = predict_action(s_next, W)
            r_next_hat = predict_reward(s_next, a_next, W)
            one_hot_states = one_hot_encoding_state(s, a)

            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                break

            W = W + alpha * (r + gamma * r_next_hat - r_hat) * one_hot_states
            state = state_next

        if episode % 100 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, W)
            print(f"episode: {episode} - score: {score}  steps: {steps}  alpha: {alpha}")
            if episode > 500:
                alpha = 0.1
                epsilon = 0.5
            if episode > 100:
                alpha = 0.05
                epsilon = 0.2
            if episode > 2000:
                alpha = 0.01
                epsilon = 0.1

    # plt.plot(MEs)
    # plt.xlabel("Kroki")
    # plt.ylabel("ME")
    # plt.show()
    print(W)
    return 0

wahadlo_uczenie()
