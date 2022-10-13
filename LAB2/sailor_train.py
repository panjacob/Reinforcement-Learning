import copy
import time
import os
import pdb
from random import random, randint

import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
from sailor_funct import environment

# file_name = 'map_small.txt'
file_name = 'map_easy.txt'
# file_name = 'map_middle.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

number_of_episodes = 4000  # number of training epizodes (multi-stage processes)
gamma = 1.0  # discount factor

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

# TRENING-----------------------------------
delta_max = 10
delta = delta_max + 1
V = np.zeros([num_of_rows, num_of_columns], dtype=float)
gamma = 1
while delta >= delta_max:
    V_pom = copy.deepcopy(V)
    delta = 0
    for x in range(num_of_rows):
        for y in range(num_of_columns):
            v_max = 0
            for action1 in [1, 2, 3, 4]:
                v_val = 0
                for action2 in [1, 2, 3, 4]:
                    state_next, reward, probability = environment([x, y], action1, reward_map)
                    v1 = reward + gamma * probability * V_pom[state_next[0], state_next[1]]
                    v2 = reward + gamma * probability * V_pom[state_next[0], state_next[1]]
                    delta = max( - V_pom[x, y])

            print(state_next, reward)
            # V_pom[x,y] = max(reward)
    # strategy = np.random.randint(1, 5, (num_of_rows, num_of_columns))
    # for episode in range(number_of_episodes):
    #     state = np.zeros([2], dtype=int)  # initial state here [1 1] but rather random due to exploration
    #     state[0] = np.random.randint(0, num_of_rows)
    #     for y in range(1000):
    #         if epsilon > random():
    #             action = randint(1, 4)
    #         else:
    #             action = strategy[state[0], state[1]]
    #
    #         state_next, reward = environment(state, action, reward_map);
    #         Q[state[0], state[1], action - 1] += alfa * (
    #                 reward + gamma * max(Q[state_next[0], state_next[1], :]) - Q[
    #             state[0], state[1], action - 1])
    #         state = state_next
    #
    #         if state[1] >= num_of_columns - 1:
    #             break
# TRENING-----------------------------------


sf.sailor_test(reward_map, V, 1000)
sf.draw(reward_map, V)
