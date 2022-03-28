import numpy as np

a = np.random.random([3,3])

a1 = np.sqrt(np.sum(a*a))

a2 = np.sqrt(np.trace(a@a.T))

print(a1, a2)



