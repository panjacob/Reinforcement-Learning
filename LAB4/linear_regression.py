import numpy as np
import matplotlib.pyplot as plt

W = np.array([0.0, 0.0])
W_len = W.shape[0]


def predict(x, W):
    return W[0] * x[0] + W[1] * x[1]


def f(x):
    return x[0] ** 2 - x[1]


X = []
for a in range(5):
    for b in range(5, 10):
        X.append([a, b])

X = np.array(X)
Y = np.array([f(x) for x in X])
n = X.shape[0]
print(X[:, 1])
#
lr = 0.001
MSE_ALL = []
for e in range(10000):
    Y_hat = np.array([predict(x, W) for x in X])
    MSE = (1 / n) * np.sum(Y - Y_hat)
    for wi in range(W_len):
        derivative_wi = (-2 / n) * np.sum(X[:, wi] * (Y - Y_hat))
        W[wi] -= lr * derivative_wi

    MSE_ALL.append(MSE)

print(W)
# plt.plot(MSE_ALL)
# plt.ylabel('some numbers')
# plt.show()
