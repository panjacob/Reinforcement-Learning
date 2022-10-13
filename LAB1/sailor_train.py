from random import random, randint

import numpy as np
import sailor_funct as sf
from sailor_funct import environment
from multiprocessing import Process, Pool

number_of_episodes = 500  # number of training epizodes (multi-stage processes)

file_name = 'map_small.txt'  # best score: [0.8, 0.4, 0.3, 2.329]
# file_name = 'map_easy.txt' # best score: [0.9, 0.2, 0.5, 6.692]
# file_name = 'map_middle.txt' # best score: [0.9, 0.2, 0.3, 6.9355]
# file_name = 'map_big.txt'  # best score: [0.7, 0.5, 0.4, -1.3065]
# file_name = 'map_spiral.txt' # best score: [0.9, 0.2, 0.6, 178.7605]

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode


# trained usability table of <state,action> pairs
# sum_of_rewards = np.zeros([number_of_episodes], dtype=float)


def train(arguments):
    gamma, alfa, epsilon = arguments
    Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)
    strategy = np.random.randint(1, 5, (num_of_rows, num_of_columns))
    for episode in range(number_of_episodes):
        state = np.zeros([2], dtype=int)  # initial state here [1 1] but rather random due to exploration
        state[0] = np.random.randint(0, num_of_rows)
        for y in range(1000):
            if epsilon > random():
                action = randint(1, 4)
            else:
                action = strategy[state[0], state[1]]

            state_next, reward = environment(state, action, reward_map);
            Q[state[0], state[1], action - 1] += alfa * (reward + gamma * max(Q[state_next[0], state_next[1], :]))
            state = state_next

            if state[1] >= num_of_columns - 1:
                break
    score = sf.sailor_test(reward_map, Q, 1000)

    # print(f"gamma: {gamma} alfa: {alfa} epsilon: {epsilon} score: {score}")
    # print('.')
    return [gamma, alfa, epsilon, score]


def test(arguments):
    gamma, alfa, epsilon = arguments
    return [gamma, alfa, epsilon, randint(0, 10)]


if __name__ == '__main__':
    # TEST 1 - dobór parametrów
    # values = []
    # for gamma in np.arange(0.6, 1, 0.05):
    #     for alfa in np.arange(0.2, 0.6, 0.05):
    #         for epsilon in np.arange(0.3, 0.7, 0.05):
    #             values.append([gamma, alfa, epsilon])
    #
    # with Pool(12) as pool:
    #     scores = [pool.map(train, values)]
    #
    # best = [-1, -1, -1, -100]
    # for score in scores[0]:
    #     if score[3] > best[3]:
    #         best = score
    #
    # print(f"best score: {str(best)}")

    #     TEST2 - test wybranych parametrów
    best = [-1, -1, -1, -100]
    for i in range(500):
        score = train([0.8, 0.4, 0.3])
        if score[3] > best[3]:
            best = score
        print(i)
    print(best)

# W tym miejscu mozna uzyc kodu z funkcji sailor_test()

# sf.sailor_test(reward_map, Q, 1000)
# sf.draw(reward_map,Q)
