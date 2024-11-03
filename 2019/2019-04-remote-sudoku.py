from itertools import product

import numpy as np
from z3 import And, Distinct, IntVector, Or, Solver, sat

grid = [
    [2, 0, 0, 0, 7, 1, 8, 3, 6],
    [0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 5, 0, 0, 5, 4, 0, 2],
    [0, 0, 0, 1, 0, 0, 5, 0, 1],
    [8, 3, 3, 0, 1, 0, 2, 4, 4],
    [3, 0, 4, 0, 0, 3, 0, 0, 0],
    [6, 0, 2, 3, 0, 0, 5, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 0],
    [7, 2, 7, 3, 1, 0, 0, 0, 3],
]


def square(k: int) -> np.ndarray:
    """Return a boolean matrix with True in the kth square."""
    i, j = divmod(k, 3)
    i0, j0 = 3 * i, 3 * j
    i1, j1 = i0 + 3, j0 + 3
    sq = np.full((9, 9), False)
    sq[i0:i1, j0:j1] = True
    return sq


remote = []
for i, j in product(range(9), repeat=2):
    dist = grid[i][j]
    if dist:
        direc = [(dist, 0), (-dist, 0), (0, dist), (0, -dist)]
        neigh = [(i + di, j + dj) for di, dj in direc]
        coord = [(i, j) for i, j in neigh if 0 <= i < 9 and 0 <= j < 9]
        remote.append((dist, coord))

# Define variables and solver
X = np.array(IntVector("x", 9**2)).reshape(9, 9)
s = Solver()

# Add constraints to the solver
s += [And(x >= 1, x <= 9) for x in X.flat]
s += [Distinct(*X[i, :]) for i in range(9)]
s += [Distinct(*X[:, j]) for j in range(9)]
s += [Distinct(*X[square(k)]) for k in range(9)]
s += [Or([X[i, j] == dist for i, j in coords]) for dist, coords in remote]

# Solve the problem
if s.check() == sat:
    m = s.model()
    xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
    ans = np.sum(xm**2 * np.where(grid, 1, 0))
    print(f"{ans = }\n", "xm = \n", xm)
else:
    print("no solution")
# ans = 1105
# grid =
# 6 9 2  5 3 7  1 8 4
# 1 3 7  8 4 2  5 6 9
# 5 8 4  1 9 6  2 3 7

# 4 5 3  7 8 9  6 1 2
# 9 1 6  4 2 3  7 5 8
# 7 2 8  6 1 5  4 9 3

# 2 7 1  9 5 8  3 4 6
# 8 6 5  3 7 4  9 2 1
# 3 4 9  2 6 1  8 7 5
