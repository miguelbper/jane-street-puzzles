"""Let m = side of the square grid = 10 k = number of regions = 17.

    n = number of steps the knight takes
    s = sum of every number in the grid
    r = sum of the numbers in a region of the grid

Then,
    s = n*(n+1) / 2
    s = 17 * r
    n <= 100
    r >= (12 + 5 + 8 + 2) + (11 + 13) = 51

Lem. r <= 2*n - 3.
Pf. Consider a region with two squares. The biggest sum that region can
have is n + (n - 3) = 2*n - 3.

Lem. If r is even, then r <= n.
Pf. Consider a region with two squares. If the knight were to visit that
region twice, then r would be odd (sum of an even and odd number). So,
the knight visits that region exactly once.

These facts taken together imply that
    n = 50
    r = 75
    s = 1275
"""

from itertools import product

import numpy as np
from codetiming import Timer
from z3 import And, Implies, IntVector, PbEq, Solver, sat

# Data
# ----------------------------------------------------------------------
# fmt: off
regions = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  1,  0,  0,  0,  0,  2,  0],
    [ 0,  0,  0,  1,  0,  0,  0,  0,  2,  0],
    [ 0,  0,  1,  1,  1,  1,  3,  2,  2,  2],
    [ 5,  0,  1,  0,  0,  4,  3,  3,  3,  3],
    [ 5,  0,  0,  0,  4,  4,  4, 10, 10,  3],
    [ 5,  6,  6,  7,  4,  9,  9, 10, 10, 11],
    [ 5, 12,  6,  7,  7,  8,  8,  8, 10, 11],
    [ 5, 12, 13,  7, 16, 16, 16, 15, 15, 11],
    [12, 12, 13, 14, 14, 14, 14, 14, 15, 11],
])

grid = np.array([
    [12, 0, 0,  0,  0, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0,  0, 0, 5, 0, 23,  0],
    [ 0, 0, 0,  0,  0, 0, 8, 0,  0,  0],
    [ 0, 0, 0, 14,  0, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
    [ 0, 2, 0,  0,  0, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0, 20, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0, 33, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0,  0, 0, 0, 0,  0, 28],
])
# fmt: on
n, _ = grid.shape
num_regions = np.max(regions) + 1
num_steps = 50
sum_region = 75


s = Solver()
X = np.array(IntVector("x", n**2)).reshape((n, n))

s += [And(x >= 0, x <= num_steps) for x in X.flat]
s += [PbEq([(x == m, 1) for x in X.flat], 1) for m in range(1, num_steps + 1)]
s += [sum(X[reg]) == sum_region for reg in regions == np.arange(num_regions).reshape(-1, 1, 1)]
s += [x == y for x, y in zip(X.flat, grid.flat) if y]

directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

for i, j in product(range(n), repeat=2):
    knight_moves = [(i + di, j + dj) for di, dj in directions if 0 <= i + di < n and 0 <= j + dj < n]

    # next
    needs_move = And(X[i, j] != 0, X[i, j] != num_steps)
    moves = [(X[i, j] + 1 == X[k, l], 1) for k, l in knight_moves]
    s += Implies(needs_move, PbEq(moves, 1) if moves else False)

    # previous
    needs_move = And(X[i, j] != 0, X[i, j] != 1)
    moves = [(X[i, j] - 1 == X[k, l], 1) for k, l in knight_moves]
    s += Implies(needs_move, PbEq(moves, 1) if moves else False)


with Timer(initial_text="Checking z3 solver"):
    check = s.check()

if check == sat:
    m = s.model()
    xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
    ans = np.sum(np.max(xm, axis=1) ** 2)
    print(f"answer = {ans}")
    print(xm)
# Elapsed time: 1.0842 seconds
# answer = 14820
# [[12  0  0  0  0  9  0  7  0  0]
#  [ 0  0 13 10  0  0  5  0 23  0]
#  [ 0 11  0 17  4  0  8  0  6  0]
#  [ 1  0  0 14  0 18  0 22  0 24]
#  [ 0  0 16  3  0 21 50 25  0  0]
#  [ 0  2  0  0 15  0 19 48  0  0]
#  [ 0 41 34  0 20 49 26  0  0 47]
#  [35 38  0 42 33 30 45  0 27  0]
#  [40  0 36  0  0 43 32 29 46  0]
#  [37  0 39  0 31  0  0 44  0 28]]
