# from random import random

import numpy as np
from matplotlib import pyplot as plt

from utilis import *
from tqdm import tqdm


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji

class PrototypesMesh:
    def __init__(self, distance):
        self.states = np.random.rand(RESOLUTION, RESOLUTION, RESOLUTION, RESOLUTION, RESOLUTION)
        self.distance = distance
        self.all_state_actions = []

    # def generate_all_state_actions(self):
    #     result = []
    #     for s0 in range(RESOLUTION):
    #         for s1 in range(RESOLUTION):
    #             for s2 in range(RESOLUTION):
    #                 for s3 in range(RESOLUTION):
    #                     for a in range(RESOLUTION):
    #                         for range()
    #                         result.append(np.array((s0, s1, s2, s3, a)))
    #     return result
    def range_val(self, value):
        min_val = max(0, value - self.distance)
        max_val = min(RESOLUTION, value + self.distance + 1)
        return [x for x in range(min_val, max_val) if x != value]

    def near(self, s, a):
        result = []
        s0, s1, s2, s3 = s
        # print()

        for x0 in self.range_val(s0):
            for x1 in self.range_val(s1):
                for x2 in self.range_val(s2):
                    for x3 in self.range_val(s3):
                        for x4 in self.range_val(a):
                            result.append([x0, x1, x2, x3, x4])

        return result


#
# class Prototype:
#     def __init__(self, state, action):
#         self.state = np.array(state + [action])


def wahadlo_uczenie(episode_count=10_000, alpha=0.1, gamma=0.9, epsilon=0.5):
    W = np.random.rand(FEATURE_COUNT, RESOLUTION)
    max_steps = 1000
    mesh = PrototypesMesh(1)

    for episode in range(episode_count):
        # state = BEGIN_STATES[-1]
        state = BEGIN_STATES[episode % BEGIN_STATES_COUNT]
        for i in range(max_steps):
            r = reward(state)
            s = encode_states(state)
            a = random_action() if random.random() < epsilon else predict_action(s, W)

            r_hat = predict_reward(s, a, W)
            state_next = wahadlo(state, a)
            s_next = encode_states(state_next)
            a_next = predict_action(s_next, W)
            r_next_hat = predict_reward(s_next, a_next, W)

            # gradient = np.zeros((FEATURE_COUNT, RESOLUTION), dtype=float)
            gradient = one_hot_encoding_state(s, a)
            near_states = mesh.near(s, a)
            for s0, s1, s2, s3, a in near_states:
                gradient = one_hot_encoding_state([s0, s1, s2, s3], a, gradient)

            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                break
            alpha = 1 / (len(near_states) + 1)
            W += alpha * (r + gamma * r_next_hat - r_hat) * gradient
            state = state_next

        if episode % 100 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, W)
            print(f"episode: {episode} - score: {score}  steps: {steps}  alpha: {alpha}")
            # if episode > 500:
            #     alpha = 0.1
            #     epsilon = 0.5
            # if episode > 100:
            #     alpha = 0.05
            #     epsilon = 0.2
            # if episode > 2000:
            #     alpha = 0.01
            #     epsilon = 0.1

    # plt.plot(MEs)
    # plt.xlabel("Kroki")
    # plt.ylabel("ME")
    # plt.show()
    print(W)
    return 0


wahadlo_uczenie()
