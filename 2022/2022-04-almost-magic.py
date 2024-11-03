import numpy as np
from codetiming import Timer
from z3 import And, Distinct, IntVector, Optimize, sat


def square(i: int, j: int) -> np.ndarray:
    """Return a boolean matrix with True in [i:i+3, j:j+3]."""
    sq = np.full((6, 6), False)
    sq[i : i + 3, j : j + 3] = True
    return sq


corners = [(0, 1), (2, 0), (3, 2), (1, 3)]
grid = np.logical_or.reduce([square(i, j) for i, j in corners])


# Variables, optimizer & objective
# ----------------------------------------------------------------------
X = np.array(IntVector("x", 6**2)).reshape((6, 6))
s = Optimize()
s.minimize(sum(X.flat))


# Constraints
# ----------------------------------------------------------------------

# Range for the variables
s += [x >= 1 for x in X[grid]]
s += [x == 0 for x in X[~grid]]

# Numbers in grid should be distinct
s += Distinct(*X[grid])

# Each square is almost magic
for i, j in corners:
    X_sq = X[i : i + 3, j : j + 3]
    rows = list(np.sum(X_sq, axis=1))
    cols = list(np.sum(X_sq, axis=0))
    diag = [np.trace(X_sq), np.trace(np.fliplr(X_sq))]
    sums = np.array(rows + cols + diag).reshape(-1, 1)
    diff = np.triu(sums - sums.T)
    diff = diff[~(diff == 0)]
    s += [And(d >= -1, d <= 1) for d in diff]

# Break symmetry, so that solution is unique (=> faster)
s += X[3, 3] < X[2, 2]
s += X[3, 3] < X[2, 3]
s += X[3, 3] < X[3, 2]

# Solve
# ----------------------------------------------------------------------
with Timer():
    sat_ = s.check()

if sat_ == sat:
    m = s.model()
    xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
    ans = xm[grid].flatten()
    print(f"sum = {np.sum(xm)}")
    print(f"ans = {ans}")
    print("xm = \n", xm)
else:
    print("No solution")
# Elapsed time: 161.2319 seconds
# sum = 470
# ans = [26, 2, 34, 29, 21, 13, 4, 18, 24, 7, 39, 16, 12, 6, 37, 23, 9, 5, 19, 10, 8, 40, 22, 11, 1, 3, 17, 14]  # noqa: E501
# xm =
# [[ 0 26  2 34  0  0]
#  [ 0 29 21 13  4 18]
#  [24  7 39 16 12  6]
#  [37 23  9  5 19 10]
#  [ 8 40 22 11  1  0]
#  [ 0  0  3 17 14  0]]
