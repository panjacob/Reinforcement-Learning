import numpy as np

#
# #
# R = [0, 1, 2, 3, 4]
# W = [0, 0, 0, 0, 0]
# Q = [0, 0, 0, 0, 0]
#
# # W = (0.3 ** i) * l_gradient ** 2
#
# for i, r in enumerate(R):
#     W[i] = (0.9 ** i) * R[i]
#
# for i, w in enumerate(W):
#     Q[i] = sum(W[i:])
#
# print(Q)

x = np.gradient([-1, 8, 8])
print(x)

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
print(a * b)
