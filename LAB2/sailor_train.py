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
# file_name = 'map_easy.txt'
# file_name = 'map_middle.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

# Najlepszy wynik dla mapy: map_small.txt - 1.909375   gamma=0.95
# Najlepszy wynik dla mapy: map_easy.txt - 5.629125   gamma=0.84
# Najlepszy wynik dla mapy: map_middle.txt - 7.3345   gamma=0.98
# Najlepszy wynik dla mapy: map_big.txt - 0.4905   gamma=0.98
# Najlepszy wynik dla mapy: map_spiral.txt - 183.246125   gamma=0.99

number_of_episodes = 4000  # number of training epizodes (multi-stage processes)


# reward_map = sf.load_data(file_name)
# num_of_rows, num_of_columns = reward_map.shape


def calculate_reward(state, reward_map, action=None):
    # num_of_rows, num_of_columns = reward_map.shape
    if action is None:
        return reward_map[state[0], state[1]]
    num_of_rows, num_of_columns = reward_map.shape
    wall_colid_reward = -1.5
    state_new = copy.deepcopy(state)
    reward = 0

    if action == 1:
        if state[1] < num_of_columns - 1:
            state_new[1] += 1
            reward += reward_map[state_new[0], state_new[1]]
        else:
            reward += wall_colid_reward
    elif action == 2:
        if state[0] > 0:
            state_new[0] -= 1
            reward += reward_map[state_new[0], state_new[1]]
        else:
            reward += wall_colid_reward
    if action == 3:
        if state[1] > 0:
            state_new[1] -= 1
            reward += reward_map[state_new[0], state_new[1]]
        else:
            reward += wall_colid_reward
    elif action == 4:
        if state[0] < num_of_rows - 1:
            state_new[0] += 1
            reward += reward_map[state_new[0], state_new[1]]
        else:
            reward += wall_colid_reward

    return reward


# num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
# sum_of_rewards = np.zeros([number_of_episodes], dtype=float)


# print(reward_map)
# print('reward', calculate_reward([0, 0], reward_map, 3))
def train(gamma, reward_map):
    probability_of_desired_action = 0.69
    delta_max = 1e-40
    delta = 1
    V = np.zeros([num_of_rows, num_of_columns], dtype=float)
    strategy = np.zeros([num_of_rows, num_of_columns], dtype=float)
    i = 0
    while delta >= delta_max:
        i += 1
        delta = 0
        if i > 10000:
            # print("STOP - MAX ITERATIONS!")
            break
        V_pom = copy.deepcopy(V)
        for x in range(num_of_rows):
            for y in range(num_of_columns - 1):
                state = [x, y]
                reward_max = -999
                strategy_max = 8
                for action_desired in [1, 2, 3, 4]:
                    reward_desired = probability_of_desired_action * calculate_reward(state, reward_map, action_desired)
                    V_desired = probability_of_desired_action * calculate_reward(state, V_pom, action_desired)
                    reward_undesired = 0
                    V_undesired = 0
                    for action_undesired in [1, 2, 3, 4]:
                        if action_desired == action_undesired:
                            continue
                        probability_undesired = 0.15 if (action_desired - action_undesired) % 2 else 0.01
                        V_undesired += probability_undesired * calculate_reward(state, V_pom, action_undesired)
                        reward_undesired += probability_undesired * calculate_reward(state, reward_map,
                                                                                     action_undesired)

                    reward = (reward_desired + reward_undesired) + gamma * (V_desired + V_undesired)
                    # actionX = {1: 'right', 2: 'up', 3: 'left', 4: 'bottom'}
                    # print('state:', f"[x:{state[1]} y:{state[0]}", 'reward', reward, 'reward_map',
                    #       calculate_reward(state, reward_map, action_desired), 'action',
                    #       actionX[action_desired], 'reward_desired:', reward_desired,
                    #       'reward_undesired:', reward_undesired, 'reward', reward)

                    if reward > reward_max:
                        reward_max = reward
                        strategy_max = action_desired
                delta = max(delta, np.abs(V_pom[x, y] - reward_max))
                V_pom[x, y] = reward_max
                strategy[x, y] = strategy_max

        V = copy.deepcopy(V_pom)
    return strategy, i


# file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
# file_name = 'map_middle.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'
for file_name in [ 'map_easy.txt', 'map_middle.txt', 'map_big.txt', 'map_spiral.txt']:
    reward_map = sf.load_data(file_name)
    num_of_rows, num_of_columns = reward_map.shape
    max_gamma = None
    max_result = -999
    for gamma in np.arange(0.6, 1, 0.01):
        strategy, i = train(gamma, reward_map)
        test = sf.sailor_test(reward_map, strategy, 4000)
        if test > max_result:
            max_result = test
            max_gamma = gamma
        # print(test, f'gamma={round(gamma, 4)} i:{i}')
    # print(f'Najlepszy wynik dla mapy: {file_name} - {max_result}   gamma={round(max_gamma, 3)}')

# strategy, _ = train()
# print(reward_map)
# test = sf.sailor_test(reward_map, strategy, 4000)
# print("TEST: ", test)
# sf.draw(reward_map, strategy)
