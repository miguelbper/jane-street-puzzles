import numpy as np
from z3 import IntVector, Optimize, sat, And
from codetiming import Timer

d = np.array([10**(4 - i) for i in range(5)])

# Variables and optimizer
X = np.array(IntVector('x', 5**2)).reshape((5, 5))
s = Optimize()

# Objective
s.maximize(sum(X.flat))

# Constraints
s += [And(0 <= x, x <= 9) for x in X.flat]
s += [(X[i, :] @ d) % (i + 1) == 0 for i in range(5)]
s += [(X[:, j] @ d) % (j + 6) == 0 for j in range(5)]

# Solve
with Timer():
    sat_ = s.check()

if sat_ == sat:
    m = s.model()
    xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
    ans = (np.sum(xm), int(''.join(map(str, xm.flat))))
    print(f'{ans = }')
    print(xm)
else:
    print('No solution')
'''
Elapsed time: 0.4171 seconds
ans = (205, 9899999998798999989689890)
[[9 8 9 9 9]
 [9 9 9 9 8]
 [7 9 8 9 9]
 [9 9 8 9 6]
 [8 9 8 9 0]]
'''
