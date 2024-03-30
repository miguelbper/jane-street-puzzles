import numpy as np
from z3 import IntVector, Solver, And, Or, Distinct, sat, Implies
from itertools import product
from codetiming import Timer

# define grids
z = 0
values = np.array([
    [z, 3, z, z, z, 7, z, z, z, z],
    [z, z, z, 4, z, z, z, z, z, z],
    [z, z, z, z, z, z, z, z, 2, z],
    [z, z, z, 1, z, z, z, z, z, z],
    [6, z, 1, z, z, z, z, z, z, z],
    [z, z, z, z, z, z, z, 3, z, 6],
    [z, z, z, z, z, z, 2, z, z, z],
    [z, 2, z, z, z, z, z, z, z, z],
    [z, z, z, z, z, z, 6, z, z, z],
    [z, z, z, z, 5, z, z, z, 2, z],
])

regions = np.array([
    [ 0,  1,  1,  1,  2,  2,  2,  2,  2,  2],
    [ 0,  0,  1,  1,  1,  2,  3,  3,  2,  2],
    [ 0,  0,  4,  4,  5,  5,  6,  3,  7,  7],
    [ 0,  0,  8,  4,  9,  6,  6,  6,  7,  7],
    [ 0,  8,  8,  9,  9, 10, 11,  6,  6,  7],
    [ 0, 12,  8, 13, 14, 10, 15, 15,  7,  7],
    [ 0, 16, 17, 13, 13, 13, 15, 20, 20, 20],
    [16, 16, 17, 18, 13, 19, 22, 21, 21, 22],
    [16, 16, 16, 18, 18, 22, 22, 21, 21, 22],
    [16, 16, 16, 16, 18, 18, 22, 22, 22, 22],
])
n_regions = np.max(regions) + 1
region_coords = [regions == r for r in range(n_regions)]
region_sizes = [np.sum(region) for region in region_coords]

n = values.shape[0]
IJ = list(product(range(n), repeat=2))

def dist(p: tuple[int, int], q: tuple[int, int]) -> int:
    '''Taxicab distance between p=(i0, j0) and q=(i1, j1).'''
    i0, j0 = p
    i1, j1 = q
    return abs(i0 - i1) + abs(j0 - j1)


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

# For each number K, the nearest K via taxicab must be exactly K cells away
for i, j in IJ:
    p = (i, j)
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
    ans = np.sum(np.prod(xm, axis=1))
    print(f'{ans = }')
    print('xm = \n', xm)
else:
    print('No solution found.')
'''
Elapsed time: 0.0941 seconds
ans = 24405360
xm =
[[ 4  3  6  5  3  7  4  9  6  5]
 [ 8 10  2  4  1  1  2  3  8  2]
 [ 9  2  3  2  1  2  5  1  2  4]
 [ 5  7  2  1  2  6  3  1  1  3]
 [ 6  3  1  1  3  2  1  4  2  7]
 [ 1  1  4  5  1  1  1  3  5  6]
 [ 3  1  2  3  2  4  2  1  2  3]
 [ 4  2  1  1  1  1  3  1  4  9]
 [ 5  8  3  4  2  1  6  2  3  8]
 [ 7  6  9 10  5  3  4  7  2  5]]
'''
