from z3 import Solver, Distinct, sat, And, Or, Implies, IntVector
from itertools import product
import numpy as np
from codetiming import Timer


# Inputs
# ----------------------------------------------------------------------
regions = np.array([
    [ 0,  0, 22,  2,  2,  3,  3,  3,  4],
    [ 0,  0,  2,  2,  6,  7,  3,  3,  4],
    [ 0,  5,  6,  6,  6,  7,  7,  4,  4],
    [ 9,  5,  6, 21,  6, 12,  7, 10, 10],
    [ 9,  9, 13, 13, 14, 14, 16, 17, 17],
    [ 9,  9, 14, 14, 14, 15, 16, 18, 17],
    [19,  9, 23, 23, 20, 20, 18, 18, 18],
    [19,  8,  8, 11, 11, 20, 20,  1, 18],
    [19, 11, 11, 11, 11,  1,  1,  1, 18],
])

values = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
])

n_regions = np.max(regions) + 1
region_coords = [regions == r for r in range(n_regions)]
region_sizes = [np.sum(region) for region in region_coords]

n = values.shape[0]
IJ = list(product(range(n), repeat=2))


def dist(p: tuple[int, int], q: tuple[int, int]) -> int:
    '''Horizontal/vertical distance from p=(i0, j0) to q=(i1, j1).'''
    i0, j0 = p
    i1, j1 = q
    di = abs(i0 - i1)
    dj = abs(j0 - j1)
    return float('inf') if di and dj else di + dj


def hcn(xm: np.ndarray, k: int) -> list[int]:
    '''List of horizontally concatenated numbers in region k.'''
    ans = []
    for row, region in zip(xm, regions):
        num_str = ''.join(map(str, row[region == k]))
        num = int(num_str) if num_str else 0
        ans.append(num)
    return ans


def answer(xm: np.ndarray) -> int:
    '''Return the answer given the grid.'''
    return sum(max(hcn(xm, k)) for k in range(n_regions))


# Variables, solver, constraints
# ----------------------------------------------------------------------
s = Solver()
X = np.array(IntVector('x', n**2)).reshape((n, n))

# Given numbers
s += [x == v for x, v in zip(X.flat, values.flat) if v]

# Fill each region with the numbers {1,...,N}, where N = region_size
for region, N in zip(region_coords, region_sizes):
    s += Distinct(*X[region])
    s += [And(1 <= x, x <= N) for x in X[region]]

# If K is in a cell, the nearest K is K cells away
for i, j in IJ:
    p = i, j
    N = region_sizes[regions[i, j]]
    for k in range(1, N + 1):
        # X[i, j] == k => no cell at distance < k has a k
        inter = np.array([0 < dist(p, q) < k for q in IJ]).reshape(n, n)
        no_close_k = And([x != k for x in X[inter]])
        # X[i, j] == k => there exists a cell at distance k with a k
        perim = np.array([dist(p, q) == k for q in IJ]).reshape(n, n)
        exists_k = Or([x == k for x in X[perim]])
        # Add implication
        s += Implies(X[i, j] == k, And(no_close_k, exists_k))

# Solve
# ----------------------------------------------------------------------
with Timer():
    sat_ = s.check()

if sat_ == sat:
    m = s.model()
    xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
    ans = answer(xm)
    print(f'{ans = }')
    print('xm = \n', xm)
else:
    print('No solution found.')
'''
Elapsed time: 0.0390 seconds
ans = 6647
xm =
[[3 4 1 1 3 4 2 3 2]
 [2 5 2 4 2 3 5 1 3]
 [1 1 3 6 1 1 2 1 4]
 [3 2 4 1 5 1 4 1 2]
 [5 4 2 1 2 5 1 1 3]
 [1 2 3 4 1 1 2 3 2]
 [1 6 1 2 4 2 5 6 4]
 [3 2 1 3 1 1 3 1 1]
 [2 4 2 6 5 4 2 3 2]]
'''
