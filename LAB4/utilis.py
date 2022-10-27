from math import pi
import numpy as np
from random import randint, random
from numpy.random import rand

RESOLUTION = 40
# Bins of from state0 to state3 and bins of F
BIN_MIN = [-10, -10, -10, -10, -10]
BIN_MAX = [10, 10, 10, 10, 10]
F_MAX_RANGE = abs(BIN_MIN[4]) + BIN_MAX[4] - 1


def best_action(state_decoded, Q):
    state = encode_states(state_decoded)
    index = np.argmax(Q[state[0], state[1], state[2], state[3], :])
    action_value = Q[state[0], state[1], state[2], state[3], index]
    return action_value, index


def random_action():
    index = randint(0, F_MAX_RANGE)
    action_value = BINS[4][index]
    return action_value, index


BINS = [
    np.linspace(BIN_MIN[0], BIN_MAX[0], num=RESOLUTION),
    np.linspace(BIN_MIN[1], BIN_MAX[1], num=RESOLUTION),
    np.linspace(BIN_MIN[2], BIN_MAX[2], num=RESOLUTION),
    np.linspace(BIN_MIN[3], BIN_MAX[3], num=RESOLUTION),
    np.linspace(BIN_MIN[4], BIN_MAX[4], num=RESOLUTION),
]


def encode_states(state):
    result = []
    for i, x in enumerate(state):
        if x >= BIN_MAX[i]:
            # print(f"BIN MAX ERROR: i={i} x={x}")
            x = BIN_MAX[i]
        if x <= BIN_MIN[i]:
            # print(f"BIN MIN ERROR: i={i} x={x}")
            x = BIN_MIN[i]
        result.append(np.digitize(x, BINS[i]) - 1)
    return result


def decode_Q(encoded):
    state = []
    for i, x in enumerate(encoded):
        state.append(BINS[i][x])
    # F = BINS[4][encoded[4]]
    return state


# Obliczenie stanu wahadla w kolejnym kroku czasowym metoda analityczna
# stan - wektor parametrow stanu w czasie t
# stann -   -||- w czasie t + dt
# F - sila dzialajaca na wozek
def wahadlo(stan, F):
    Fmax, krokcalk, g, tar, masawoz, masawah, drw = wah_glob()

    if F > Fmax:
        F = Fmax
    if F < -Fmax:
        F = -Fmax

    hh = krokcalk * 0.5;
    momwoz = masawoz * drw;
    momwah = masawah * drw;
    cwoz = masawoz * g;
    cwah = masawah * g;

    sx = np.sin(stan[0]);
    cx = np.cos(stan[0]);
    c1 = masawoz + masawah * sx * sx;
    c2 = momwah * stan[1] * stan[1] * sx;
    c3 = tar * stan[3] * cx;

    stanpoch = np.zeros(stan.size)

    stanpoch[0] = stan[1];
    stanpoch[1] = ((cwah + cwoz) * sx - c2 * cx + c3 - F * cx) / (drw * c1);
    stanpoch[2] = stan[3];
    stanpoch[3] = (c2 - cwah * sx * cx - c3 + F) / c1;
    stanh = np.zeros(stan.size)
    for i in range(4):
        stanh[i] = stan[i] + stanpoch[i] * hh;

    sx = np.sin(stanh[0]);
    cx = np.cos(stanh[0]);
    c1 = masawoz + masawah * sx * sx;
    c2 = momwah * stanh[1] * stanh[1] * sx;
    c3 = tar * stanh[3] * cx;

    stanpochh = np.zeros(stan.size)
    stanpochh[0] = stanh[1];
    stanpochh[1] = ((cwah + cwoz) * sx - c2 * cx + c3 - F * cx) / (drw * c1);
    stanpochh[2] = stanh[3];
    stanpochh[3] = (c2 - cwah * sx * cx - c3 + F) / c1;
    stann = np.zeros(stan.size)
    for i in range(4):
        stann[i] = stan[i] + stanpochh[i] * krokcalk;
    if stann[0] > np.pi:
        stann[0] = stann[0] - 2 * pi;
    if stann[0] < -np.pi:
        stann[0] = stann[0] + 2 * pi;

    return stann


def wah_glob():
    Fmax = 1000
    krokcalk = 0.05
    g = 9.8135
    tar = 0.02
    masawoz = 10
    masawah = 20
    drw = 20
    return Fmax, krokcalk, g, tar, masawoz, masawah, drw


def save_states():
    Fmax, krokcalk, g, tar, masawoz, masawah, drw = wah_glob()
    pli = open('historia.txt', 'w')
    pli.write("Fmax = " + str(Fmax) + "\n")
    pli.write("krokcalk = " + str(krokcalk) + "\n")
    pli.write("g = " + str(g) + "\n")
    pli.write("tar = " + str(tar) + "\n")
    pli.write("masawoz = " + str(masawoz) + "\n")
    pli.write("masawah = " + str(masawah) + "\n")
    pli.write("drw = " + str(drw) + "\n")
    return pli


def wahadlo_test(state_begin, Q):
    file = save_states()

    reward_sum = 0
    step_count = 0
    begin_step_count, lparam = state_begin.shape
    for episode in range(begin_step_count):
        # Wybieramy stan poczatkowy:
        # stan = rand(1,4).*[pi/1.5 pi/1.5 20 20] - [pi/3 pi/3 10 10]; % met.losowa
        nr_stanup = episode
        state = state_begin[nr_stanup, :]

        step = 0
        suma_nagrod_epizodu = 0
        czy_przewrocenie_wahadla = 0
        while (step < 1000) & (czy_przewrocenie_wahadla == 0):
            step = step + 1

            F, _ = best_action(state, Q)

            nowystan = wahadlo(state, F)

            czy_przewrocenie_wahadla = (abs(nowystan[0]) >= np.pi / 2)
            R = reward(state, nowystan, F)
            suma_nagrod_epizodu = suma_nagrod_epizodu + R

            file.write(
                str(episode + 1) + "  " + str(state[0]) + "  " + str(state[1]) + "  " + str(state[2]) + "  " + str(
                    state[3]) + "  " + str(F) + "\n")

            state = nowystan

        reward_sum = reward_sum + suma_nagrod_epizodu / begin_step_count
        step_count = step_count + step
        # print("w %d epizodzie suma nagrod = %g, liczba krokow = %d" % (epizod, suma_nagrod_epizodu, krok))

    print("srednia suma nagrod w epizodzie = %g" % (reward_sum))
    print("srednia liczba krokow ustania wahadla = %g" % (step_count / begin_step_count))

    file.close()


def reward(stan, nowystan, F):
    kara_za_odchylenie = nowystan[0] ** 2 + 0.25 * nowystan[1] ** 2 + 0.0025 * nowystan[2] ** 2 + 0.0025 * nowystan[
        3] ** 2
    kara_za_przewrocenie = (abs(nowystan[0]) >= np.pi / 2) * 1000
    # ..............................................
    # ..............................................
    return -(kara_za_odchylenie + kara_za_przewrocenie)


BEGIN_STATES = np.array(
    [
        [np.pi / 6, 0, 0, 0],
        [0, np.pi / 3, 0, 0],
        [0, 0, 10, 0], [0, 0, 0, 10],
        [np.pi / 12, np.pi / 6, 0, 0],
        [np.pi / 12, -np.pi / 6, 0, 0],
        [-np.pi / 12, np.pi / 6, 0, 0],
        [-np.pi / 12, -np.pi / 6, 0, 0],
        [np.pi / 12, 0, 0, 0],
        [0, 0, -10, 10]],
    dtype=float)
BEGIN_STATES_COUNT, lparam = BEGIN_STATES.shape
