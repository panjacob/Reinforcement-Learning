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
file_name = 'map_middle.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

number_of_episodes = 4000  # number of training epizodes (multi-stage processes)

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape


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


num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)


# print(reward_map)
# print('reward', calculate_reward([0, 0], reward_map, 3))
def train(gamma, alpha):
    probability_of_desired_action = 0.69
    delta_max = 0.0000000000000000000000000000000001
    delta = 999
    V = np.zeros([num_of_rows, num_of_columns], dtype=float)
    strategy = np.zeros([num_of_rows, num_of_columns], dtype=float)
    # gamma = 0.2  # discount factor
    i = 0
    while delta >= delta_max:
        i += 1
        if i > 10000:
            print("STOP - MAX ITERATIONS!")
            break
        V_pom = copy.deepcopy(V)
        for x in range(num_of_rows):
            for y in range(num_of_columns - 1):
                state = [x, y]
                reward_current = calculate_reward(state, reward_map)
                reward_max = -999
                strategy_max = 8
                for action_desired in [1, 2, 3, 4]:
                    reward_undesired = 0
                    for action_undesired in [1, 2, 3, 4]:
                        if action_desired == action_undesired:
                            continue
                        probability_undesired = 0.15 if (action_desired - action_undesired) % 2 else 0.01
                        reward_undesired += probability_undesired * calculate_reward(state, V_pom, action_undesired)

                    reward_desired = probability_of_desired_action * calculate_reward(state, V_pom, action_desired)
                    reward = reward_current + gamma * (reward_desired + reward_undesired) + alpha * calculate_reward(
                        state, reward_map, action_desired)
                    # map_reward_next = alpha * calculate_reward(state, reward_map, action_desired)
                    # if map_reward_next < 0:
                    #     reward += map_reward_next
                    # actionX = {1: 'right', 2: 'up', 3: 'left', 4: 'bottom'}
                    # print('state:', f"[x:{state[1]} y:{state[0]}", 'reward', reward, 'reward_map',
                    #       calculate_reward(state, reward_map, action_desired), 'action',
                    #       actionX[action_desired], 'reward_desired:', reward_desired,
                    #       'reward_undesired:', reward_undesired, 'reward', reward)

                    if reward > reward_max:
                        reward_max = reward
                        strategy_max = action_desired
                V_pom[x, y] = reward_max
                strategy[x, y] = strategy_max

        delta = abs(V_pom.sum() - V.sum())
        V = copy.deepcopy(V_pom)
        # print('delta', delta)
    return strategy, i


max_params = None
max_result = -999
for gamma in np.arange(0.7, 0.9, 0.1):
    for alpha in np.arange(0.6, 1.2, 0.1):
        strategy, i = train(gamma, alpha)
        # print(V)
        # print(strategy)
        test = sf.sailor_test(reward_map, strategy, 4000)
        if test > max_result:
            max_result = test
            max_params = [gamma, alpha]
        print("TEST: ", test, f'gamma={round(gamma, 4)} alpha={round(alpha, 4)} i:{i}')
print('MAX_PARAMS: ', max_params, "SCORE: ", max_result)

# strategy, _ = train()
# print(reward_map)
# test = sf.sailor_test(reward_map, strategy, 4000)
# print("TEST: ", test)
# sf.draw(reward_map, strategy)
