from random import random

import numpy as np
from matplotlib import pyplot as plt

from utilis4 import *


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji


def wahadlo_uczenie(gamma=0.99, epsilon=0.1, episode_count=1_000, minibatch_size=8):
    W = np.random.rand(FEATURE_COUNT)
    max_steps = 1000
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

        if episode % 100 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, W)
            print(f"episode: {episode} - score: {score}  steps: {steps}  lr:{lr}")
    return 0


wahadlo_uczenie()
