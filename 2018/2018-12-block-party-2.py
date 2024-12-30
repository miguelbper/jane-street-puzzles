from itertools import permutations

import numpy as np
from codetiming import Timer
from z3 import (
    And,
    ArithRef,
    BoolRef,
    Distinct,
    IntVector,
    Not,
    Optimize,
    Or,
    PbEq,
    Solver,
)

# fmt: off
regions = np.array([
    [ 0,  1,  1,  2,  2,  3,  3],
    [ 0,  0,  1,  5,  2,  3,  7],
    [ 4,  4,  5,  5,  6,  7,  7],
    [ 8,  4, 10, 10,  6,  6, 12],
    [ 8,  8,  9, 10, 11, 12, 12],
    [13,  9,  9, 15, 11, 11, 14],
    [13, 13, 15, 15, 15, 14, 14],
])
# fmt: on

n, _ = regions.shape
bigger_region = 15


# Lower bound
# ----------------------------------------------------------------------

max_digit = 7
max_range = (max_digit - 1) * (max_digit - 2) + 1

s = Optimize()
X = np.array(IntVector("x", max_digit**2)).reshape((max_digit, max_digit))

s += [x >= 0 for x in X.flat]
s += [x == 0 for x in X[np.tril_indices(max_digit)]]
s += [x == 0 for x in X[:2, :].flat]
s += np.sum(X) == 16

I, J = np.indices((n, n))
K = np.arange(max_range)
mask = K.reshape(-1, 1, 1) == I * J
B = np.sum(np.where(mask, X * mask, 0), axis=(1, 2))
C = np.pad(np.sum(X, axis=0) + np.sum(X, axis=1), (0, max_range - max_digit), mode="constant")
A = B + C
s += [a <= n for a in A if isinstance(a, ArithRef)]

objective = np.sum(I * X * J)
s.minimize(objective)

with Timer(initial_text="Finding lower bound..."):
    s.check()
m = s.model()
xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
score = np.sum(I * xm * J)
print(f"{score = }")
print("xm = \n", xm, sep="")

pairs = [(a, b, k) for (a, b), k in np.ndenumerate(xm) if k]
for a, b, k in pairs:
    print(f"{k} * {(a, b)}")
# Finding lower bound...
# Elapsed time: 0.0550 seconds
# score = np.int64(210)
# xm =
# [[0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0]
#  [0 0 0 0 1 3 3]
#  [0 0 0 0 4 2 1]
#  [0 0 0 0 0 2 0]
#  [0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0]]
# 1 * (2, 4)
# 3 * (2, 5)
# 3 * (2, 6)
# 4 * (3, 4)
# 2 * (3, 5)
# 1 * (3, 6)
# 2 * (4, 5)

# Upper bound / Solution
# ----------------------------------------------------------------------

s = Solver()
X = np.array(IntVector("x", n**2)).reshape((n, n))

regions_values = np.unique(regions).reshape(-1, 1, 1)
regions_levels = regions_values == regions
s += [Distinct(*row) for row in X]
s += [Distinct(*col) for col in X.T]
s += [Distinct(*X[reg]) for reg in regions_levels]


def maximizes_region(i: np.int64, j: np.int64) -> BoolRef:
    region_num = regions[i, j]
    region_mask = regions == region_num
    return And([X[i, j] >= x for x in X[region_mask]])


I, J, K, L = np.indices((n, n, n, n))
adjacent = np.abs(K - I) + np.abs(L - J) == 1
for i, j, k, l in np.argwhere(adjacent):
    s += Not(And([maximizes_region(i, j), maximizes_region(k, l)]))


def array_equal(xs: list[ArithRef], ts: list[int]) -> BoolRef:
    return And([x == t for x, t in zip(xs, ts)])


def array_equal_permutation(xs: list[ArithRef], ts: list[int]) -> BoolRef:
    return Or([array_equal(xs, list(ps)) for ps in permutations(ts, len(ts))])


def region_equals(r: np.int64, pair: tuple[int, int]) -> BoolRef:
    a, b = pair
    extra_element = r == bigger_region
    xs = [x for x in X[regions == r].flat]
    ts = [1, a, b, a * b] if extra_element else [a, b, a * b]
    return array_equal_permutation(xs, ts)


for a, b, k in pairs:
    s += PbEq([(region_equals(r, (a, b)), 1) for r in np.unique(regions)], k)


with Timer(initial_text="\nFinding upper bound / solution..."):
    s.check()
m = s.model()
xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
score = np.sum(np.max(xm * regions_levels, axis=(1, 2)))
print(f"{score = }")
print("xm = \n", xm, sep="")
# Finding upper bound / solution...
# Elapsed time: 11.4722 seconds
# score = np.int64(210)
# xm =
# [[ 4 10  5  6  3 12  2]
#  [12  3  2  5 18  6  4]
#  [ 2  6 20  4  5  3 12]
#  [ 5 12  6  2 20  4  3]
#  [10  2  3 12  4  5 15]
#  [ 3  4 12  1  2  8  5]
#  [15  5  4  3 12  2 10]]
