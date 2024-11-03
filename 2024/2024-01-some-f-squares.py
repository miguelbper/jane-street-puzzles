# Imports
# ----------------------------------------------------------------------
from functools import reduce
from itertools import product
from operator import mul

import numpy as np
from codetiming import Timer
from scipy.ndimage import label, sum_labels
from tqdm import tqdm
from z3 import And, BoolRef, Implies, IntVector, ModelRef, Or, Solver, sat

# Given grid
# ----------------------------------------------------------------------
# fmt: off
regions = np.array([
    [ 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
    [ 0, 0,  0,  1,  1,  1,  1,  2,  2,  2,  3,  4,  4,  4,  4,  3, 3],
    [ 0, 5,  5,  5,  1,  1,  1,  2,  6,  2,  3,  4,  3,  3,  4,  7, 3],
    [ 0, 5,  5,  1,  1,  1,  1,  6,  6,  2,  3,  3,  3,  4,  4,  7, 3],
    [ 0, 5,  8,  1,  6,  6,  6,  6,  9, 10, 10,  3,  3,  4,  7,  7, 3],
    [ 0, 5,  8,  8,  6,  8,  9,  9,  9, 11, 10,  3,  3,  4,  4,  7, 7],
    [ 0, 5, 12,  8,  8,  8,  9,  9,  9, 11, 10,  3,  3, 13,  4,  7, 7],
    [ 0, 5, 12, 12,  8,  9,  9, 11, 11, 11, 10,  3, 10, 13,  7,  7, 7],
    [ 0, 5, 12, 12, 12,  9,  9, 11, 10, 10, 10, 10, 10, 13, 13,  7, 7],
    [ 0, 5, 12, 12, 12, 12, 11, 11, 14, 14, 10, 13, 10, 13,  7,  7, 7],
    [ 0, 5, 12, 12, 12, 11, 11, 15, 15, 14, 14, 13, 13, 13, 13, 13, 7],
    [ 0, 5, 12, 12, 11, 11, 15, 15, 15, 16, 14, 14, 17, 17,  7, 13, 7],
    [ 0, 5, 12, 15, 15, 15, 15, 15, 16, 16, 16, 14, 14, 17,  7,  7, 7],
    [ 0, 5, 15, 15, 15, 15, 15, 15, 16, 18, 16, 17, 17, 17, 17,  7, 7],
    [ 0, 5,  5, 15,  0,  0, 19, 16, 16, 18, 17, 17, 18, 18, 17, 17, 7],
    [ 0, 0,  5,  5,  0, 19, 19, 18, 18, 18, 18, 18, 18, 18,  7,  7, 7],
    [ 0, 0,  0,  0,  0,  0, 19, 19, 19, 19, 19, 19, 19, 18,  7,  7, 7],
])
# fmt: on
rows = [14, 24, 24, 39, 43, 0, 22, 23, 29, 28, 34, 36, 29, 26, 26, 24, 20]
cols = [13, 20, 22, 28, 30, 36, 35, 39, 49, 39, 39, 0, 23, 32, 23, 17, 13]
n = regions.shape[0]
num_regions = np.max(regions) + 1

Board = np.ndarray[(n, n), int]


# Missing sum of row / col
# ----------------------------------------------------------------------
"""
Let
    x = missing col sum
    y = missing row sum
    s = sum of all numbers in the grid
    k = sum of a region

Then
    s = x + sum(rows) = x + 458
    s = y + sum(cols) = y + 441
    s = k * num_regions = k * 20

Possibilities:
     k   s  x  y
    23 460  2 19
    24 480 22 39 <-
    25 500 42 59
    26 520 62 79
"""
rows[5] = 39
cols[11] = 22
region_sum = 24


# Utility functions
# ----------------------------------------------------------------------
def areas(xm: Board) -> int:
    """Product of the areas of the unfilled regions."""
    mat = np.where(xm == 0, 1, 0)
    labels, k = label(mat)
    area = sum_labels(mat, labels, index=range(1, k + 1))
    return int(reduce(mul, area))


def evaluate_vars(m: ModelRef, vars: np.ndarray) -> np.ndarray:
    """Evaluate variables in a z3 model."""
    return np.vectorize(lambda x: m.evaluate(x).as_long())(vars)


def print_arr(arr: np.ndarray, name: str) -> None:
    """Prints a numpy 2D array where each element is a str."""
    print(f"{name} = ", end="")
    for i, row in enumerate(arr):
        initial_spaces = (3 + len(name)) * " " if i else ""
        row = " ".join(np.char.strip(row, chars="'"))
        print(initial_spaces + "[" + row + "]")


# Solver + variables + constraints
# ----------------------------------------------------------------------
s = Solver()
X = np.array(IntVector("x", n**2)).reshape((n, n))

# Ranges for each variable
s += [And(x >= 0, x <= 3) for x in X.flat]

# The sum of the numbered squares inside each region must be the same
s += [sum(X[regions == r]) == region_sum for r in range(num_regions)]

# A number outside the grid = sum of squares in row or column
s += [sum(X[i, :]) == r for i, r in enumerate(rows)]
s += [sum(X[:, j]) == c for j, c in enumerate(cols)]


# Pentominos
# ----------------------------------------------------------------------
# Phrase condition as:
# X[i, j] == k  =>  there exists a k-pentomino region containing (i, j)


def pentomino(k: int) -> np.ndarray:
    """Returns a pentomino with number m."""
    nums = k * np.ones((k, k), dtype=int)
    zero = np.zeros((k, k), dtype=int)
    arr = np.block([[nums, nums, zero], [zero, nums, nums], [zero, nums, zero]])
    return arr


def pentominos(k: int) -> list[np.ndarray]:
    """Returns all rotations/reflections of the m-pentomino."""
    ans = []
    for rotation in range(4):
        ans.append(np.rot90(pentomino(k), rotation))
        ans.append(np.rot90(np.fliplr(pentomino(k)), rotation))
    return ans


def in_pentomino(i: int, j: int, k: int) -> BoolRef:
    """z3 bool: True iff (i, j) is in a pentomino with number m."""
    conditions = []
    side = 3 * k

    # loop over pentominos with number m (all rotations and reflections)
    for p in pentominos(k):
        # loop over translations of the pentomino with m in (i, j)
        a_min = max(0, i - side + 1)
        a_max = min(n - side, i)
        b_min = max(0, j - side + 1)
        b_max = min(n - side, j)
        for a, b in product(range(a_min, a_max + 1), range(b_min, b_max + 1)):
            if p[i - a, j - b] != k:
                continue

            # condition: p[u, v] == k => X[a + u, b + v] == k
            cell_values = []
            for u, v in product(range(side), repeat=2):
                if p[u, v]:
                    cell_values.append(X[a + u, b + v] == k)

            conditions.append(And(cell_values))

    return Or(conditions)


desc = "Pentomino constraints"
for i, j in tqdm(product(range(n), repeat=2), total=n**2, desc=desc):
    for k in range(1, 4):
        s += Implies(X[i, j] == k, in_pentomino(i, j, k))


# Solve
# ----------------------------------------------------------------------
with Timer(initial_text="Checking z3 solver"):
    check = s.check()

if check == sat:
    m = s.model()
    xm = evaluate_vars(m, X)
    xm_str = np.vectorize(lambda x: "." if x == 0 else str(x))(xm)
    print_arr(xm_str, "xm")
    print(f"answer = {areas(xm)}")
"""
Pentomino constraints: 100%|██████████| 289/289 [01:14<00:00,  3.88it/s]
Checking z3 solver
Elapsed time: 4.1584 seconds
xm = [. 1 . . 1 1 3 3 3 . . . . . 1 1 .]
     [1 1 1 1 1 . 3 3 3 . . 2 2 2 2 1 1]
     [1 . . . 1 . 3 3 3 2 2 2 2 2 2 1 .]
     [3 3 3 3 3 3 3 3 3 2 2 . . 2 2 2 2]
     [3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2]
     [3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 . .]
     [. 1 . 3 3 3 . 2 2 2 2 . . 2 2 . .]
     [. 1 1 3 3 3 . 2 2 2 2 . . 2 2 . .]
     [1 1 1 3 3 3 2 2 3 3 3 . . 2 2 . .]
     [. 1 1 1 . . 2 2 3 3 3 2 2 2 2 2 2]
     [. 1 2 2 2 2 2 2 3 3 3 2 2 2 2 2 2]
     [1 1 2 2 2 2 2 2 3 3 3 3 3 3 . 2 2]
     [. 1 1 1 2 2 . . 3 3 3 3 3 3 . 2 2]
     [. 1 1 1 2 2 . . 3 3 3 3 3 3 1 . .]
     [. 1 1 1 1 3 3 3 3 3 3 . 1 1 1 1 .]
     [. . 1 1 . 3 3 3 3 3 3 1 1 1 . 1 .]
     [. . 1 . . 3 3 3 3 3 3 . . 1 . . .]
answer = 346816512
"""
