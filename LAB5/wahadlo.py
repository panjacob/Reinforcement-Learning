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
        # self.W = np.zeros((FEATURE_COUNT, RESOLUTION), dtype=float)
        # self.max_dist_sqrt = sqrt(distance)
        self.max_dist = distance

    # def near(self, s, a):
    #     x = np.array(s + [a])
    #     result = []
    #     for s0 in range(self.W.shape[1]):
    #         if s0 - self.max_dist_sqrt >= x[0] or a + self.max_dist_sqrt <= x[4]:
    #             continue
    #         for s1 in range(self.W.shape[1]):
    #             if s1 - self.max_dist_sqrt >= x[1] or a + self.max_dist_sqrt <= x[4]:
    #                 continue
    #             for s2 in range(self.W.shape[1]):
    #                 if s2 - self.max_dist_sqrt >= x[2] or a + self.max_dist_sqrt <= x[4]:
    #                     continue
    #                 for s3 in range(self.W.shape[1]):
    #                     if s3 - self.max_dist_sqrt >= x[3] or a + self.max_dist_sqrt <= x[4]:
    #                         continue
    #                     for a in range(self.W.shape[1]):
    #                         if a - self.max_dist_sqrt >= x[4] or a + self.max_dist_sqrt <= x[4]:
    #                             continue
    #                         y = np.array([s0, s1, s2, s3, a])
    #                         dist = np.linalg.norm(x - y)
    #                         if dist <= self.max_dist:
    #                             result.append(y)
    #     return result

    def gen_range(self, x):
        range_min = max(0, x - self.max_dist)
        range_max = min(RESOLUTION, x + self.max_dist + 1)
        ranges = []
        for i in range(range_min, range_max):
            if i == x:
                continue
            ranges.append(i)
        # print(x, ranges)
        return ranges

    def near(self, s, a):
        result = []
        for s0 in self.gen_range(s[0]):
            for s1 in self.gen_range(s[1]):
                for s2 in self.gen_range(s[2]):
                    for s3 in self.gen_range(s[3]):
                        for a in self.gen_range(a):
                            result.append([s0, s1, s2, s3, a])

        return result


def wahadlo_uczenie(episode_count=10_000, alpha=0.1, gamma=0.7, epsilon=0):
    max_steps = 1000
    mesh = PrototypesMesh(1)

    for episode in range(episode_count):
        state = BEGIN_STATES[episode % BEGIN_STATES_COUNT]
        for i in range(max_steps):
            r = reward(state)
            s = encode_states(state)
            a = random_action() if random.random() < epsilon else predict_action(s, mesh.W, text='now')
            # print(s, a)
            action = BINS[4][a]
            # print(action)

            r_hat = predict_reward(s, a, mesh.W)
            state_next = wahadlo(state, action)
            s_next = encode_states(state_next)
            a_next = predict_action(s_next, mesh.W, text='next')
            r_next_hat = predict_reward(s_next, a_next, mesh.W)

            # gradient = np.zeros((FEATURE_COUNT, RESOLUTION), dtype=float)
            gradient = one_hot_encoding_state(s, a)
            # near_states = mesh.near(s, a)
            #
            # for s0, s1, s2, s3, a in near_states:
            #     gradient = one_hot_encoding_state([s0, s1, s2, s3], a, gradient)

            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                break
            # alpha = 1 / ((len(near_states) + 1) * 1)
            mesh.W += alpha * (r + gamma * r_next_hat - r_hat) * gradient
            state = state_next

        if episode % 100 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, mesh.W)
            print(f"episode: {episode} - score: {score}  steps: {steps}  alpha: {epsilon}")

            if episode > 500:
                epsilon = 0.8
            if episode > 1000:
                epsilon = 0.2
            if episode > 2000:
                epsilon = 0.1

    print(mesh.W)
    return 0


wahadlo_uczenie()
