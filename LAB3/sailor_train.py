import copy
import time
import os
import pdb
from math import floor
from random import random, randint
from multiprocessing import Process, Pool
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
from sailor_funct import environment

maps = ['map_small.txt', 'map_easy.txt', 'map_middle.txt', 'map_big.txt', 'map_spiral.txt']


def findBestAction(state, Q):
    best_position = -1
    max_reward = -999
    for x in [0, 1, 2, 3]:
        reward = Q[state[0], state[1], x]
        if reward > max_reward:
            max_reward = reward
            best_position = x
    return best_position + 1


def findMaxValue(state, Q):
    best_position = -1
    max_reward = -999
    for x in [0, 1, 2, 3]:
        reward = Q[state[0], state[1], x]
        if reward > max_reward:
            max_reward = reward
            best_position = x
    return best_position + 1


def calculate_reward(state, reward_map, action=None):
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


def calc_average_reward(state_action_reward):
    total_reward = 0
    for _, _, reward in state_action_reward:
        total_reward += reward
    return total_reward / len(state_action_reward)


def train(arugments):
    epsilon, alpha, gamma, reward_map = arugments
    num_of_rows, num_of_columns = reward_map.shape
    Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)
    for i in range(5000):
        state = [0, 0]
        while state[1] <= num_of_columns - 2:
            action = randint(1, 4) if epsilon > random() else findBestAction(state, Q)
            state_new, reward = environment(state, action, reward_map)
            Q[state[0], state[1], action - 1] = Q[state[0], state[1], action - 1] + alpha * \
                                                (reward + gamma * max(Q[state_new[0], state_new[1], :]) - Q[
                                                    state[0], state[1], action - 1])
            state = state_new
    print('.', end='')
    return Q, (epsilon, alpha, gamma)


def find_parameters():
    for file_name in maps:
        best_test = -999
        best_epsilon = -999
        best_alpha = -999
        best_gamma = -999
        for epsilon in np.arange(0.1, 1, 0.1):
            for alpha in np.arange(0.1, 1, 0.1):
                for gamma in np.arange(0.1, 1, 0.1):
                    reward_map = sf.load_data(file_name)
                    num_of_rows, num_of_columns = reward_map.shape
                    Q1 = train(epsilon, alpha, gamma, reward_map)
                    Q2 = train(epsilon, alpha, gamma, reward_map)
                    Q3 = train(epsilon, alpha, gamma, reward_map)

                    test1 = sf.sailor_test(reward_map, Q1, 1000)
                    test2 = sf.sailor_test(reward_map, Q2, 1000)
                    test3 = sf.sailor_test(reward_map, Q3, 1000)

                    test = min(test1, test2, test3)

                    if test > best_test:
                        best_epsilon = epsilon
                        best_gamma = gamma
                        best_alpha = alpha
                        best_test = test
                    # if test > 6:
                    #     print(round(epsilon, 2), round(alpha, 2), round(gamma, 2), round(test, 4))
        print(file_name, round(best_epsilon, 3), round(best_alpha, 3), round(best_gamma, 3), round(best_test, 4))


def test_parameters():
    epsilon = 0.2
    alpha = 0.1
    gamma = 0.8
    file_name = maps[1]
    reward_map = sf.load_data(file_name)
    Q = train(epsilon, alpha, gamma, reward_map)
    test = sf.sailor_test(reward_map, Q, 1000)
    print("TEST: ", test)
    sf.draw(reward_map, Q)


def find_parameters_fast(reward_map):
    parameters = []
    for epsilon in np.arange(0.1, 1, 0.1):
        for alpha in np.arange(0.1, 1, 0.1):
            for gamma in np.arange(0.1, 1, 0.1):
                parameters.append([epsilon, alpha, gamma, reward_map])

    print('Training')
    with Pool(12) as pool:
        Qs_and_parameters = pool.map(train, parameters)

    test_Qs = []
    for Q, parameter in Qs_and_parameters:
        test_Qs.append([reward_map, Q, 1000, parameter])

    print('\nTesting')
    with Pool(12) as pool:
        results = pool.map(sf.sailor_test, test_Qs)

    best = -100
    best_params = None
    for score, params in results:
        if score > best:
            best = score
            best_params = params
    print("\n", best, best_params)

# 0 4.057 (0.5, 0.2, 0.7)
# 1 7.418 (0.2, 0.1, 0.9)
# 2 7.764 (0.2, 0.3, 0.6)
if __name__ == "__main__":
    reward_map = sf.load_data(maps[3])
    find_parameters_fast(reward_map)
