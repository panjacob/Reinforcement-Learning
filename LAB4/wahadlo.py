from utilis import *


# episode_count = 100_000_000
# alpha - szybkość uczenia
# epsilon - współczynnik eksploarcji

def wahadlo_uczenie(alpha=0.2, epsilon=0.1, episode_count=1_000_000):
    Q = np.zeros([RESOLUTION, RESOLUTION, RESOLUTION, RESOLUTION, RESOLUTION], dtype=float)

    for episode in range(episode_count):
        state_start = episode % BEGIN_STATES_COUNT
        state = BEGIN_STATES[state_start, :]

        for i in range(episode_count):
            F, F_index = random_action() if random() < epsilon else best_action(state, Q)
            state_new = wahadlo(state, F)
            R = reward(state, state_new, F)
            s = encode_states(state)
            best_action_next, _ = best_action(state_new, Q)
            # Q[s[0], s[1], s[2], s[3], F_index] = R

            update = Q[s[0], s[1], s[2], s[3], F_index] + alpha * \
                     (R + 0.8 * best_action_next - Q[s[0], s[1], s[2], s[3], F_index])
            Q[s[0], s[1], s[2], s[3], F_index] = update

            state = state_new

            if abs(state_new[0]) >= (np.pi / 2):
                # print('Upadek po ', i)
                break

        # co jakis czas test z wygenerowaniem historii do pliku:
        if episode % 1000 == 0:
            print((episode / episode_count) * 100, "%")
            wahadlo_test(BEGIN_STATES, Q)


wahadlo_uczenie()
