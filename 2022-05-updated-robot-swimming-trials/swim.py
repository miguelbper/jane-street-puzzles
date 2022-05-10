from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

# compute p
e = lambda p: 1 - p - (1 - p/8)**24 + (7*p/8)**24
p = fsolve(e, 0.98)[0]
print('{:6f}'.format(p))

# plot probability that R1 wins as a function of p
x_min = 0.98
x_max = 1
x = np.linspace(x_min, x_max, 100)
y = (1 - (1 - x/8)**24 + (7*x/8)**24)/(3*x)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.axhline(y=1/3, color='gray', linewidth=0.2)
ax.axvline(x=p, color='gray', linewidth=0.2)
ax.set_xticks([x_min, p, x_max])
ax.set_xticklabels(['0.98', '{:6f}'.format(p), '1'])
ax.set_yticks([1/3])
ax.set_yticklabels(['1/3'])
ax.set_xlabel('p')
ax.set_ylabel('probability')
plt.show()