# from random import random
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt

from utilis import *
from tqdm import tqdm


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji

class PrototypesMesh:
    def __init__(self, distance):
        self.W = np.random.rand(FEATURE_COUNT, RESOLUTION)
        self.max_dist_sqrt = sqrt(distance)
        self.max_dist = distance

    def near(self, s, a):
        x = np.array(s + [a])
        result = []
        for s0 in range(self.W.shape[1]):
            if s0 - self.max_dist_sqrt >= x[0] or a + self.max_dist_sqrt <= x[4]:
                continue
            for s1 in range(self.W.shape[1]):
                if s1 - self.max_dist_sqrt >= x[1] or a + self.max_dist_sqrt <= x[4]:
                    continue
                for s2 in range(self.W.shape[1]):
                    if s2 - self.max_dist_sqrt >= x[2] or a + self.max_dist_sqrt <= x[4]:
                        continue
                    for s3 in range(self.W.shape[1]):
                        if s3 - self.max_dist_sqrt >= x[3] or a + self.max_dist_sqrt <= x[4]:
                            continue
                        for a in range(self.W.shape[1]):
                            if a - self.max_dist_sqrt >= x[4] or a + self.max_dist_sqrt <= x[4]:
                                continue
                            y = np.array([s0, s1, s2, s3, a])
                            dist = np.linalg.norm(x - y)
                            if dist <= self.max_dist:
                                result.append(y)
        return result


def wahadlo_uczenie(episode_count=10_00, alpha=0.1, gamma=0.9, epsilon=0.5):
    max_steps = 1000
    mesh = PrototypesMesh(1)

    for episode in tqdm(range(episode_count)):
        state = BEGIN_STATES[episode % BEGIN_STATES_COUNT]
        for i in range(max_steps):
            r = reward(state)
            s = encode_states(state)
            a = random_action() if random.random() < epsilon else predict_action(s, mesh.W)

            r_hat = predict_reward(s, a, mesh.W)
            state_next = wahadlo(state, a)
            s_next = encode_states(state_next)
            a_next = predict_action(s_next, mesh.W)
            r_next_hat = predict_reward(s_next, a_next, mesh.W)

            # gradient = np.zeros((FEATURE_COUNT, RESOLUTION), dtype=float)
            gradient = one_hot_encoding_state(s, a)
            near_states = mesh.near(s, a)

            for s0, s1, s2, s3, a in near_states:
                gradient = one_hot_encoding_state([s0, s1, s2, s3], a, gradient)

            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                break
            alpha = 1 / (len(near_states) + 1)
            mesh.W += alpha * (r + gamma * r_next_hat - r_hat) * gradient
            state = state_next

        if episode % 10 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, mesh.W)
            print(f"episode: {episode} - score: {score}  steps: {steps}  alpha: {alpha}")

    print(mesh.W)
    return 0


wahadlo_uczenie()
