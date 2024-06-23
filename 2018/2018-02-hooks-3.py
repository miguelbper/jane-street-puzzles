from itertools import product
from functools import reduce
from operator import mul
from typing import Optional
from z3 import (Solver, And, Or, Not, Implies, If, PbEq, sat, BoolRef,
                IntVector, ModelRef)
from scipy.ndimage import label, sum_labels
from pprint import pprint
from codetiming import Timer
import numpy as np


# Parameters
# ----------------------------------------------------------------------

clues = np.array([
    [0, 49, 63,  0, 18, 42, 63, 54, 0],
    [0, 35, 42, 18, 18,  0, 36, 63, 0],
    [0, 56,  0, 32, 40, 15, 16, 25, 0],
    [0, 40, 32, 40, 10, 12,  0, 56, 0],
])
n = clues.shape[1]
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


# Variables, solver & constraints
# ----------------------------------------------------------------------
s = Solver()
X = np.array(IntVector('x', n**2)).reshape((n, n))  # nums in grid
O = IntVector('o', n)                               # hook orientation

# Ranges for each variable
s += [And(0 <= x, x <= n) for x in X.flat]
s += [And(0 <= o, o <= 3) for o in O]

# Every 2-by-2 region must contain at least one unfilled square
for i, j in product(range(n - 1), repeat=2):
    dirs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    s += Or([X[i + di, j + dj] == 0 for di, dj in dirs])

# There are d d's in the board
s += [PbEq([(x == d, 1) for x in X.flat], d) for d in range(1, n + 1)]

# The number in the kth hook is n - k
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
    s += Implies(in_hook(i, j, k), Or(X[i, j] == 0, X[i, j] == n - k))

# Number outside the grid = product of the first 2 numbers visible
def visible_product(side: int, clue_idx: int, clue: int) -> BoolRef:
    '''z3 bool expression: True <=> the product of the first 2 numbers
    visible from the side is 'clue'.'''
    # Array visible from side at clue_dix
    dim = side % 2
    X_arr = np.take(X, indices=clue_idx, axis=dim)
    if side in [0, 3]:
        X_arr = X_arr[::-1]

    def case_first_visible(k: int, l: int) -> BoolRef:
        '''Case where first 2 visible nums are at k and l.'''
        first_k_empty = And([x == 0 for x in X_arr[:k]])
        k_to_l_empty = And([x == 0 for x in X_arr[k+1:l]])

        possibilities = []
        for x in range(1, 10):
            y, rem = divmod(clue, x)
            if not rem:
                possibility_xy = And(X_arr[k] == x, X_arr[l] == y)
                possibility_yx = And(X_arr[k] == y, X_arr[l] == x)
                possibilities.append(Or(possibility_xy, possibility_yx))

        # product_is_clue = X_arr[k] * X_arr[l] == clue
        product_is_clue = Or(possibilities)

        return And(first_k_empty, k_to_l_empty, product_is_clue)

    return Or([case_first_visible(k, l) for l in range(n) for k in range(l)])

for side in range(4):
    for clue_idx, clue in enumerate(clues[side]):
        if clue:
            s += visible_product(side, clue_idx, clue)

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
Elapsed time: 1.2200 seconds
ans = 20736
board =
array([[0, 0, 0, 0, 9, 9, 0, 9, 0],
       [8, 7, 7, 0, 0, 7, 0, 7, 0],
       [0, 0, 6, 6, 0, 6, 6, 7, 9],
       [8, 0, 4, 0, 0, 1, 0, 0, 9],
       [8, 5, 4, 3, 2, 2, 0, 0, 9],
       [0, 5, 0, 3, 0, 3, 6, 7, 0],
       [0, 0, 0, 4, 0, 4, 0, 7, 9],
       [0, 5, 0, 5, 5, 0, 6, 0, 9],
       [0, 8, 8, 8, 0, 0, 8, 8, 9]])
'''