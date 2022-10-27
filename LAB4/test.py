import numpy as np

RESOLUTION = 20
# Bins of from state0 to state3 and bins of F
BIN_MIN = [-10, -10, -10, -10, -10]
BIN_MAX = [10, 10, 10, 10, 10]
F_MAX_RANGE = abs(BIN_MIN[4]) + BIN_MAX[4] - 1
BINS = [
    np.linspace(BIN_MIN[0], BIN_MAX[0], num=RESOLUTION),
    np.linspace(BIN_MIN[1], BIN_MAX[1], num=RESOLUTION),
    np.linspace(BIN_MIN[2], BIN_MAX[2], num=RESOLUTION),
    np.linspace(BIN_MIN[3], BIN_MAX[3], num=RESOLUTION),
    np.linspace(BIN_MIN[4], BIN_MAX[4], num=RESOLUTION),
]


def encode_Q(state):
    result = []
    for i, x in enumerate(state):
        if x >= BIN_MAX[i] or x < BIN_MIN[i]:
            print(f"BIN MAX ERROR: i={i} x={x}")
            x = BIN_MAX[i]
        if x <= BIN_MIN[i]:
            print(f"BIN MIN ERROR: i={i} x={x}")
            x = BIN_MIN[i]
        result.append(np.digitize(x, BINS[i]))
    return result


state = [-10.5, 10, 0, 22]
encoded = encode_Q(state)
print(encoded)
