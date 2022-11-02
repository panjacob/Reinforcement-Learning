from LAB4.old.utilis import *


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji


class Minibatch:
    def __init__(self, size=8):
        self.size = size
        self.D = [([0, 0, 0, 0], 0, 0.0, [0, 0, 0, 0])] * self.size
        self.i = 0

    def append(self, event):
        self.D[self.i] = event
        self.i = (self.i + 1) % self.size

    def get(self):
        random_index = randint(0, self.size - 1)
        return self.D[random_index]


def wahadlo_uczenie(gamma=0.99, epsilon=0.1, episode_count=10_000, minibatch_size=8):
    Q = np.random.rand(RESOLUTION[0], RESOLUTION[1], RESOLUTION[2], RESOLUTION[3], RESOLUTION[4])
    max_steps = 1000
    for episode in range(episode_count):
        isTerminal = False
        state = BEGIN_STATES[0]
        minibatch = Minibatch(minibatch_size)

        # Initialize minibatch
        for i in range(minibatch_size):
            action, action_index = random_action()
            s = encode_states(state)
            state_next = wahadlo(state, action)
            sn = encode_states(state_next)
            r = reward(state_next)
            minibatch.append((s, action_index, r, sn))
            state = state_next

        # Learning loop
        for i in range(max_steps):
            s = encode_states(state)
            action, action_index = random_action() if random() < epsilon else calc_best_action(s, Q)
            state_next = wahadlo(state, action)
            sn = encode_states(state_next)
            r = reward(state_next)
            minibatch.append((s, action_index, r, sn))

            s_j, action_index_j, r_j, sn_j = minibatch.get()

            if abs(state_next[0]) >= np.pi / 2 or abs(state_next[2]) > BIN_MAX[2]:
                isTerminal = True

            best_action_j, best_index_j = calc_best_action(sn_j, Q)
            best_score_j = calc_action_score(s, best_index_j, Q)
            y_j = isTerminal * r_j + gamma * best_score_j

            loss = (y_j - Q[s_j[0], s_j[1], s_j[2], s_j[3], action_index_j]) ** 2

            print(Q[s_j[0], s_j[1], s_j[2], s_j[3], action_index_j], ' + ', loss)
            Q[s_j[0], s_j[1], s_j[2], s_j[3], action_index_j] += loss

            state = state_next

        if episode % 1000 == 0:
            score, steps = wahadlo_test(BEGIN_STATES, Q)
            progress = round((episode / episode_count) * 100, 2)
            print(f"{progress}% - score: {score}  steps: {steps} ")

    return wahadlo_test(BEGIN_STATES, Q)


# best_steps = 0
# best_steps_params = ()
# best_score = -9999
# best_score_params = ()
# for alpha in np.arange(0.1, 1, 0.2):
#     for gamma in np.arange(0.1, 1, 0.2):
#         for epsilon in np.arange(0.1, 1, 0.2):
#             score, steps = wahadlo_uczenie(alpha, gamma, epsilon)
#             if steps > best_steps:
#                 best_steps = steps
#                 print('best steps: ', steps, (alpha, gamma, epsilon))
#             if score > best_score:
#                 best_score = score
#                 print('best score: ', score, (alpha, gamma, epsilon))

wahadlo_uczenie()
