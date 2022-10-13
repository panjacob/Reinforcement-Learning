import copy
import time
import os
import pdb
from random import random, randint

import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
from LAB2.sailor_funct import calculate_reward
from sailor_functxd import environment

# file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
file_name = 'map_middle.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

number_of_episodes = 4000  # number of training epizodes (multi-stage processes)
gamma = 1.0  # discount factor

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

# TRENING-----------------------------------
probability1 = 0.69
delta_max = 10
delta = delta_max + 1
V = np.zeros([num_of_rows, num_of_columns], dtype=float)
strategy = np.zeros([num_of_rows, num_of_columns], dtype=float)
while delta >= delta_max:
    V_pom = copy.deepcopy(V)
    delta = 0
    for x in range(num_of_rows):
        for y in range(num_of_columns):
            state = [x, y]
            reward_max = -999
            strategy_max = -1
            for action1 in [1, 2, 3, 4]:
                reward_rest = 0

                for action2 in [1, 2, 3, 4]:
                    if action1 == action2:
                        continue
                    probability2 = 0.01 if (action1 - action2 % 2) else 0.15
                    reward_rest += probability2 * calculate_reward(state, action1, reward_map)

                reward1 = probability1 * calculate_reward(state, action1, reward_map)
                reward = reward1 + reward_rest
                if reward > reward_max:
                    reward_max = reward
                    strategy_max = action1
            V[x, y] = reward_max
            strategy[x, y] = strategy_max
print(V)
print(strategy)

sf.sailor_test(reward_map, strategy, 1000)
sf.draw(reward_map, V)
