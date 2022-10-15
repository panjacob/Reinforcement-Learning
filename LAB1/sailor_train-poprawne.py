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
def findBestAction(state, Q):
    best_position = -1
    max_reward = -999
    for x in [0, 1, 2, 3]:
        reward = Q[state[0], state[1], x]
        if reward > max_reward:
            max_reward = reward
            best_position = x
    return best_position + 1


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
def calc_average_reward(state_action_reward):
    total_reward = 0
    for _, _, reward in state_action_reward:
        total_reward += reward
    return total_reward / len(state_action_reward)


def train(epsilon, reward_map):
    Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)
    max_reward = 0
    for i in range(1_000_000):
        state = [0, 0]
        state_action_reward = []
        while state[1] <= num_of_columns - 2:
            action = randint(1, 4) if epsilon > random() else findBestAction(state, Q)
            state_new, reward = environment(state, action, reward_map)
            state_action_reward.append((state, action, reward))
            state = state_new
        average_reward = calc_average_reward(state_action_reward)
        if average_reward > max_reward:
            max_reward = average_reward
            for state, action, reward in state_action_reward:
                Q[state[0], state[1], action - 1] = reward
    return Q


# file_name = 'map_small.txt'
file_name = 'map_easy.txt'
# file_name = 'map_middle.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'
# for epsilon in np.arange(0, 1, 0.1):
for epsilon in [0, 1, 0.1]:
    reward_map = sf.load_data(file_name)
    num_of_rows, num_of_columns = reward_map.shape
    Q = train(epsilon, reward_map)
    print(epsilon)
    sf.sailor_test(reward_map, Q, 1000)
    sf.draw(reward_map, Q)

# test = sf.sailor_test(reward_map, strategy, 1000)
# print("TEST: ", test)
# sf.draw(reward_map, strategy)
