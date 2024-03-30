from z3 import Solver, Or, sat, IntVector, PbEq
from itertools import product
import numpy as np

row_sum = [26, 42, 11, 22, 42, 36, 29, 32, 45]
col_sum = [31, 19, 45, 16,  5, 47, 28, 49, 45]

n = len(row_sum)
grid = list(product(range(n), repeat=2))
hook = [[(i, j) for i, j in grid if max(i, j) + 1 == k] for k in range(n+1)]

# Variables
X = np.array(IntVector('x', n**2)).reshape(n, n)

# Solver and constraints
s = Solver()
s += [Or(x == 0, x == max(i, j) + 1) for (i, j), x in np.ndenumerate(X)]
s += [row_sum[i] == sum(X[i, :]) for i in range(n)]
s += [col_sum[j] == sum(X[:, j]) for j in range(n)]
s += [PbEq([(X[i, j] == k, 1) for i, j in hook[k]], k) for k in range(1, n+1)]

# Solution
if s.check() == sat:
    m = s.model()
    xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
    shaded = (np.indices((n, n)).sum(axis=0) + 1) % 2
    answer = np.sum(xm * shaded)
    print(f'{answer = }\nxm = \n{xm}')
else:
    print('no solution')
'''
ans = 158
xm =
[[1 0 3 0 0 6 7 0 9]
 [2 2 3 0 5 6 7 8 9]
 [3 0 0 0 0 0 0 8 0]
 [4 4 4 4 0 6 0 0 0]
 [5 5 5 5 0 6 7 0 9]
 [0 0 6 0 0 6 7 8 9]
 [7 0 7 7 0 0 0 8 0]
 [0 8 8 0 0 8 0 8 0]
 [9 0 9 0 0 9 0 9 9]]
'''