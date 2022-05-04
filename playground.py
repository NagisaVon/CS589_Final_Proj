import numpy as np


a = np.array([[1, 4],
       [2, 5],
       [3, 6]])

b = np.array([1, 2, 3])

print(np.vstack((a.T, b)).T)


