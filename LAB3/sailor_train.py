import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 4000                   # number of training epizodes (multi-stage processes) 
gamma = 1.0                                 # discount factor


#file_name = 'map_small.txt'
file_name = 'map_easy.txt'
#file_name = 'map_middle.txt'
#file_name = 'map_big.txt'
#file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

# W tym miejscu mozna uzyc kodu z funkcji sailor_test()

sf.sailor_test(reward_map, Q, 1000)
sf.draw(reward_map,Q)
