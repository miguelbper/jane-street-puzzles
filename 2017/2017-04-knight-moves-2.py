from functools import reduce
from itertools import product
from operator import mul

import numpy as np
from codetiming import Timer
from numpy.typing import NDArray
from scipy.ndimage import label, sum_labels
from z3 import And, Implies, IntVector, PbEq, Solver, sat

# fmt: off
regions = np.array([
    [0, 0, 0, 0,  1,  1,  1,  1,  1,  1],
    [0, 2, 2, 0,  1,  1,  1,  1,  1,  3],
    [0, 2, 4, 4,  4,  4,  4,  3,  3,  3],
    [0, 2, 2, 4,  4,  4,  4,  4,  5,  3],
    [6, 2, 2, 2,  2,  2,  4,  7,  5,  5],
    [6, 6, 8, 2, 10,  7,  7,  7,  5,  5],
    [6, 8, 8, 9, 10, 10, 10,  7,  7,  7],
    [6, 8, 9, 9, 10, 11, 10,  7, 10, 10],
    [6, 8, 8, 9, 11, 11, 10, 10, 10, 10],
    [6, 9, 9, 9,  9, 11, 11, 10, 10, 10],
], dtype=int)

steps = np.array([
    [ 1, 0,  0,  0,  0,  0, 43, 0,  0,  0],
    [ 0, 0,  0,  0,  0,  0,  0, 0,  0, 13],
    [ 0, 4,  0,  0,  0,  0,  0, 0,  0, 40],
    [ 0, 0, 46,  0,  0,  7,  0, 0,  0,  0],
    [ 0, 0,  0,  0,  0, 16,  0, 0,  0, 10],
    [ 0, 0,  0,  0, 28,  0,  0, 0, 34, 31],
    [49, 0,  0,  0,  0,  0,  0, 0,  0,  0],
    [ 0, 0,  0, 19,  0,  0,  0, 0,  0,  0],
    [ 0, 0, 25,  0,  0,  0,  0, 0, 37,  0],
    [ 0, 0, 22,  0,  0,  0,  0, 0,  0,  0],
], dtype=int)
# fmt: on

n, _ = regions.shape


def knight_moves(i: int, j: int) -> list[tuple[int, int]]:
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    ans = []
    for di, dj in directions:
        x, y = i + di, j + dj
        if not (0 <= x < n and 0 <= y < n):
            continue
        if regions[x, y] == regions[i, j]:
            continue
        ans.append((x, y))
    return ans


def areas(xm: NDArray[np.int32]) -> int:
    """Product of the areas of the unfilled regions."""
    mat = np.where(xm == 0, 1, 0)
    labels, k = label(mat)
    area = sum_labels(mat, labels, index=range(1, k + 1))
    return int(reduce(mul, area))


# m = num steps
# x = num steps per row/col
# y = num steps per region <= 5 (smallest region has 5 cells)
# m = 10 * x
# m = 12 * y
# m >= 46
# Only possibility is: x = 6, y = 5 => m = 60

s = Solver()
X = np.array(IntVector("x", n**2)).reshape((n, n))

regs = [X[regions == r] for r in range(12)]

s += [And(x >= 0, x <= 60) for x in X.flat]
s += [x == s for x, s in zip(X.flat, steps.flat) if s != 0]
s += [PbEq([(x == s, 1) for x in X.flat], 1) for s in range(1, 61)]
s += [PbEq([(x != 0, 1) for x in row], 6) for row in X]
s += [PbEq([(x != 0, 1) for x in col], 6) for col in X.T]
s += [PbEq([(x != 0, 1) for x in reg], 5) for reg in regs]

for i, j in product(range(n), repeat=2):
    needs_move = And(X[i, j] != 0, X[i, j] != 60)
    moves = [(X[i, j] + 1 == X[k, l], 1) for k, l in knight_moves(i, j)]
    s += Implies(needs_move, PbEq(moves, 1) if moves else False)


with Timer(initial_text="Solving..."):
    sol = s.check()
if sol == sat:
    m = s.model()
    xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
    ans = areas(xm)
    print(f"Answer: {ans}")
    print(xm)
# Solving...
# Elapsed time: 0.4167 seconds
# Answer: 17280
# [[ 1  0  5 60  0  0 43 14 41  0]
#  [ 0 59  2 45  6  0  8  0  0 13]
#  [ 0  4  0  0  0 44 15 42  9 40]
#  [58  0 46  3  0  7  0  0 12 33]
#  [47  0  0  0  0 16 29 32 39 10]
#  [ 0 57 48  0 28  0  0 11 34 31]
#  [49 20  0 56 17  0  0 30  0 38]
#  [ 0 23  0 19 52 27 54 35  0  0]
#  [21 50 25  0 55 18  0  0 37  0]
#  [24  0 22 51 26 53 36  0  0  0]]
