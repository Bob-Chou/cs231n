import numpy as np
A = np.arange(1, 7).reshape(2, 3)
print('A=\n', A)
# B = np.array([1, 2, 3])
C = np.array([1, 2])
# print(A + B)
# print(A + C[:, np.newaxis])
print(A[np.arange(A.shape[0]), C])
print(np.maximum(np.zeros(A.shape), 1 + A - A[np.arange(A.shape[0]), C][:, np.newaxis]))