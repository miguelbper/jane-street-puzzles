from itertools import permutations, product
from pprint import pprint

import numpy as np
from codetiming import Timer
from z3 import And, BoolRef, If, Implies, IntVector, ModelRef, Or, PbEq, Solver, sat

# Parameters
# ----------------------------------------------------------------------
# fmt: off
row_labels = [45, 44,  4, 48,  7, 14, 47, 43, 33]
col_labels = [36,  5, 47, 35, 17, 30, 21, 49, 45]
# fmt: on
n = len(row_labels)

Board = np.ndarray[(n, n), int]
Coord = tuple[int, int]


# Utility functions
# ----------------------------------------------------------------------
def evaluate_vars(m: ModelRef, vars: np.ndarray) -> np.ndarray:
    """Evaluate variables in a z3 model."""
    return np.vectorize(lambda x: m.evaluate(x).as_long())(vars)


def maximal_product(xm: Board) -> int:
    """Largest product that can be achieved."""
    prd = lambda p: np.prod([xm[i, j] for j, i in enumerate(p)])
    return max(map(prd, permutations(range(n))))


# Variables, solver & constraints
# ----------------------------------------------------------------------
X = np.array(IntVector("x", n**2)).reshape((n, n))  # nums in grid
O = IntVector("o", n)  # hook orientation

s = Solver()

# Ranges for each variable
s += [And(x >= 0, x <= n) for x in X.flat]
s += [And(o >= 0, o <= 3) for o in O]

# There are d d's in the board
s += [PbEq([(x == d, 1) for x in X.flat], d) for d in range(1, n + 1)]


# The number in the kth hook is n - k
def in_hook(i: int, j: int, k: int) -> BoolRef:
    """z3 bool expression: True <=> (i, j) is in the k-th hook."""
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

# The sum of nums in a row/col is the corresponding label
s += [sum(X[i, :]) == row_labels[i] for i in range(n)]
s += [sum(X[:, j]) == col_labels[j] for j in range(n)]


# Wrap solver in a function
def solve(s: Solver) -> Board | None:
    if s.check() == sat:
        m = s.model()
        xm = evaluate_vars(m, X)
        return xm
    return None


# Solve problem
# ----------------------------------------------------------------------
with Timer():
    xm = solve(s)
    ans = maximal_product(xm)
print(f"ans = {ans}\nboard = ")
pprint(xm)
# Elapsed time: 2.9695 seconds
# ans = 17418240
# board =
# array([[9, 0, 9, 0, 0, 9, 0, 9, 9],
#        [0, 0, 5, 5, 5, 5, 7, 8, 9],
#        [0, 0, 0, 4, 0, 0, 0, 0, 0],
#        [6, 5, 4, 3, 3, 3, 7, 8, 9],
#        [0, 0, 4, 2, 1, 0, 0, 0, 0],
#        [0, 0, 4, 0, 2, 0, 0, 8, 0],
#        [6, 0, 6, 6, 6, 6, 0, 8, 9],
#        [7, 0, 7, 7, 0, 7, 7, 8, 0],
#        [8, 0, 8, 8, 0, 0, 0, 0, 9]])
