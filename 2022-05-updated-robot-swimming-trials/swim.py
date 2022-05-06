from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

# find lower bound for p
e = lambda p: 1 - p - (1 - p/8)**24
p = fsolve(e, 1)
print(p)

# plot function of p
# x = np.linspace(0.95223967,1,100)
# y = (1 - (1 - x/8)**24)/x
# fig = plt.figure()
# plt.plot(x, y)
# plt.show()