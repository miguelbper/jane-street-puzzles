from itertools import product
from functools import reduce
from operator import mul
from typing import Optional
from z3 import (Solver, And, Or, Not, Implies, If, Distinct, PbEq, sat,
                BoolRef, IntVector, ModelRef)
from scipy.ndimage import label, sum_labels
from pprint import pprint
from codetiming import Timer
import numpy as np


# Parameters
# ----------------------------------------------------------------------
regions = np.array([
    [1,  0,  0,  0,  2,  3,  3,  4,  4],
    [1,  0,  5,  5,  2, 14,  3,  4,  7],
    [1,  8,  8,  5,  5, 14, 14,  7,  7],
    [1,  8,  9,  9,  5, 14, 10, 11, 11],
    [1,  9,  9, 12, 12, 12, 10, 13, 13],
    [1,  6,  9,  9,  9, 12, 10, 10, 13],
    [1,  6,  6, 15, 16, 12, 17, 17, 18],
    [1,  1, 15, 15, 16, 12, 17, 17, 18],
    [1, 15, 15, 16, 16, 12, 12, 18, 18],
])
n = regions.shape[0]

total_sum = sum([x**2 for x in range(1, n + 1)])
num_regions = regions.max() + 1
region_sum = total_sum // num_regions

Board = np.ndarray[(n, n), int]
Coord = tuple[int, int]


# Utility functions
# ----------------------------------------------------------------------
def neighbours(i: int, j: int) -> list[Coord]:
    ''' Return the coordinates of the neighbours of (i, j). '''
    direc = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neigh = [(i + di, j + dj) for di, dj in direc]
    return [(x, y) for x, y in neigh if 0 <= x < n and 0 <= y < n]


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


# Variables
# ----------------------------------------------------------------------
X = np.array(IntVector('x', n**2)).reshape((n, n))  # nums in grid
O = IntVector('o', n)                               # hook orientation
D = IntVector('d', n)                               # digit of each hook


# Solver & constraints
# ----------------------------------------------------------------------
s = Solver()

# Ranges for each variable
s += [And(0 <= x, x <= n) for x in X.flat]
s += [And(0 <= o, o <= 3) for o in O]
s += [And(1 <= d, d <= n) for d in D]
s += Distinct(D)
s += X[0, 2] == 8

# Every 2-by-2 region must contain at least one unfilled square
for i, j in product(range(n - 1), repeat=2):
    dirs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    s += Or([X[i + di, j + dj] == 0 for di, dj in dirs])

# There are d d's in the board
s += [PbEq([(x == d, 1) for x in X.flat], d) for d in range(1, n + 1)]

# The number in the kth hook is D[k]
def in_hook(i: int, j: int, k: int) -> BoolRef:
    '''z3 bool expression: True <=> (i, j) is in the k-th hook.'''
    i0 = sum(If(Or(o == 0, o == 3), 1, 0) for o in O[:k])
    j0 = sum(If(Or(o == 0, o == 1), 1, 0) for o in O[:k])
    i1 = i0 + n - k
    j1 = j0 + n - k
    row = If(Or(O[k] == 0, O[k] == 3), i0, i1 - 1)
    col = If(Or(O[k] == 0, O[k] == 1), j0, j1 - 1)
    in_row = And(i == row, j0 <= j, j < j1)
    in_col = And(j == col, i0 <= i, i < i1)
    return Or(in_row, in_col)

for i, j, k in product(range(n), repeat=3):
    s += Implies(in_hook(i, j, k), Or(X[i, j] == 0, X[i, j] == D[k]))

# The sum of the values in each of the connected regions must be the same
s += [sum(X[regions == r]) == region_sum for r in range(num_regions)]

# Optional Constraint: know that the last two hooks are 1 and 2
s += D[-1] == 1
s += D[-2] == 2

# Optional Constraint: square is filled => a neighbor is filled (connectivity)
for i, j in product(range(n), repeat=2):
    neigh = neighbours(i, j)
    s += Implies(X[i, j] != 0, Or([X[a, b] != 0 for a, b in neigh]))

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
    pprint(xm)
else:
    print('No solution found.')

'''
Elapsed time: 2.2973 seconds
ans = 680
board =
array([[5, 7, 8, 0, 6, 0, 6, 0, 6],
       [5, 0, 0, 9, 9, 9, 9, 9, 6],
       [5, 7, 8, 4, 0, 4, 0, 9, 0],
       [0, 0, 0, 4, 2, 2, 3, 9, 6],
       [0, 0, 8, 4, 0, 1, 0, 9, 0],
       [0, 0, 0, 3, 0, 0, 3, 9, 6],
       [0, 7, 8, 8, 8, 0, 8, 0, 8],
       [0, 0, 7, 0, 7, 0, 7, 0, 7],
       [0, 0, 0, 0, 0, 5, 5, 0, 0]])
'''
