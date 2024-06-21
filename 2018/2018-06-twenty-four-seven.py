from z3 import Solver, IntVector, PbEq, Or, sat, Not, And, ModelRef
import numpy as np
from itertools import product
from typing import Optional
from scipy.ndimage import label, sum_labels
from codetiming import Timer
from operator import mul
from functools import reduce


# Parameters
# ----------------------------------------------------------------------
grid = np.array([
    [0, 4, 0, 0, 0, 0, 0],
    [0, 0, 6, 3, 0, 0, 6],
    [0, 0, 0, 0, 0, 5, 5],
    [0, 0, 0, 4, 0, 0, 0],
    [4, 7, 0, 0, 0, 0, 0],
    [2, 0, 0, 7, 4, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
])
n = grid.shape[0]
Board = np.ndarray[(n, n), int]


# Utility functions
# ----------------------------------------------------------------------
def connected(xm: Board) -> bool:
    ''' True iff the board is connected. '''
    return label(xm)[1] <= 1


def areas(xm: Board) -> int:
    ''' Product of the areas of the unfilled regions. '''
    mat = np.where(xm == 0, 1, 0)
    labels, k = label(mat)
    area = sum_labels(mat, labels, index=range(1, k + 1))
    return int(reduce(mul, area))


def evaluate_vars(m: ModelRef, vars: np.ndarray) -> np.ndarray:
    ''' Evaluate variables in a z3 model. '''
    return np.vectorize(lambda x: m.evaluate(x).as_long())(vars)


# Variables, solver, constraints
# ----------------------------------------------------------------------
X = np.array(IntVector('x', n**2)).reshape(n, n)
s = Solver()

# Given numbers
s += [x == v for x, v in zip(X.flat, grid.flat) if v]

# Grid contains one 1, two 2’s, etc., up to seven 7’s
s += [PbEq([(x == k, 1) for x in X.flat], k) for k in range(1, 8)]

# Each row and column must contain exactly 4 numbers which sum to 20
for i in range(n):
    s += sum(X[i, :]) == 20
    s += sum(X[:, i]) == 20
    s += PbEq([(x != 0, 1) for x in X[i, :]], 4)
    s += PbEq([(x != 0, 1) for x in X[:, i]], 4)

# Every 2-by-2 subsquare must contain at least one empty cell
for i, j in product(range(n - 1), repeat=2):
    dirs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    s += Or([X[i + di, j + dj] == 0 for di, dj in dirs])

# Wrap solver in a function which also checks connectivity
def solve(s: Solver) -> Optional[Board]:
    while s.check() == sat:
        m = s.model()
        xm = evaluate_vars(m, X)
        if connected(xm):
            return xm
        else:
            s += Not(And([x == y for x, y in zip(X.flat, xm.flat)]))
    return None


# Solve problem
# ----------------------------------------------------------------------
with Timer():
    xm = solve(s)

if xm is not None:
    ans = areas(xm)
    print(f'ans = {ans}\nboard = ')
    print(xm)
else:
    print('No solution found.')
'''
Elapsed time: 0.5000 seconds
ans = 240
board =
[[7 4 3 0 6 0 0]
 [0 0 6 3 5 0 6]
 [0 0 5 0 5 5 5]
 [0 3 6 4 0 0 7]
 [4 7 0 0 0 7 2]
 [2 0 0 7 4 7 0]
 [7 6 0 6 0 1 0]]
'''
