import numpy as np
a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(a)
b=np.array([1, 2])
print(b.reshape(2, -1, -1))

