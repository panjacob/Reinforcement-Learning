import numpy as np

x = np.zeros(5)
y = np.array([1, 1, 1, 0, 0])
dist = np.linalg.norm(x - y)
print(dist, x, y)
