from functools import reduce
from itertools import product
from operator import mul

import numpy as np
from codetiming import Timer
from numpy.typing import NDArray
from scipy.ndimage import label, sum_labels
from z3 import And, Implies, IntVector, ModelRef, Not, Or, PbEq, Solver, sat

# fmt: off
nums = np.array([
    [18, 19, 30, 10, 16, 11, 12, 0],
    [35, 36, 27, 20, 33, 19, 29, 0],
    [ 6, 23,  4, 37, 25, 36, 26, 0],
    [13, 14, 12,  2,  1,  5,  7, 0],
])
# fmt: on
_, n = nums.shape


def evaluate_vars(m: ModelRef, vars: np.ndarray) -> np.ndarray:
    """Evaluate variables in a z3 model."""
    return np.vectorize(lambda x: m.evaluate(x).as_long())(vars)


def areas(xm: NDArray[np.int32]) -> int:
    """Product of the areas of the unfilled regions."""
    mat = np.where(xm == 0, 1, 0)
    labels, k = label(mat)
    area = sum_labels(mat, labels, index=range(1, k + 1))
    return int(reduce(mul, area))


def solutions(num_steps: int) -> list[np.ndarray]:
    s = Solver()
    X = np.array(IntVector("x", n**2)).reshape((n, n))

    s += [And(x >= 0, x <= num_steps) for x in X.flat]
    s += [PbEq([(x == m, 1) for x in X.flat], 1) for m in range(1, num_steps + 1)]

    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    for i, j in product(range(n), repeat=2):
        needs_move = And(X[i, j] != 0, X[i, j] != num_steps)
        knight_moves = [(i + di, j + dj) for di, dj in directions if 0 <= i + di < n and 0 <= j + dj < n]
        moves = [(X[i, j] + 1 == X[k, l], 1) for k, l in knight_moves]
        s += Implies(needs_move, PbEq(moves, 1) if moves else False)

    for k in range(4):
        for num, xs in zip(nums[k], np.rot90(X, k)):
            if num:
                s += Or([And([xs[j] == num * (j == i) for j in range(i + 1)]) for i in range(n)])

    sols = []
    while s.check() == sat:
        m = s.model()
        xm = evaluate_vars(m, X)
        sols.append(xm)
        s += Not(And([x == y for x, y in zip(X.flat, xm.flat)]))

    return sols


with Timer(initial_text="Computing all solutions..."):
    sols = []
    for num_steps in range(37, 99):
        sols_num_steps = solutions(num_steps)
        sols.extend(sols_num_steps)
        print(f"{num_steps = }, {len(sols_num_steps) = }")
        if not sols_num_steps:
            break
print(f"{len(sols) = }")
# Computing all solutions...
# num_steps = 37, len(sols_num_steps) = 12
# num_steps = 38, len(sols_num_steps) = 6
# num_steps = 39, len(sols_num_steps) = 17
# num_steps = 40, len(sols_num_steps) = 23
# num_steps = 41, len(sols_num_steps) = 28
# num_steps = 42, len(sols_num_steps) = 15
# num_steps = 43, len(sols_num_steps) = 20
# num_steps = 44, len(sols_num_steps) = 10
# num_steps = 45, len(sols_num_steps) = 7
# num_steps = 46, len(sols_num_steps) = 0
# Elapsed time: 6.4956 seconds


best = min(sols, key=lambda x: areas(x))
best_score = areas(best)
print(f"{best_score = }")
print(best)
# best_score = 144
# [[18 29  0 33 20 27  0 35]
#  [ 0  0 19 28 39 34  0 26]
#  [30 17 32  9 44 21 36  0]
#  [ 0 10 15 40  0 38 25  0]
#  [16 31  0 43  8  3 22 37]
#  [11 14 41  2  0 24  7  4]
#  [ 0  0 12  0 42  5  0 23]
#  [13  0  0  0  1  0  0  6]]
