from itertools import product

import numpy as np
from scipy.ndimage import label
from z3 import And, BoolRef, IntVector, ModelRef, Not, Or, PbEq, Solver, sat

# Parameters
# ----------------------------------------------------------------------
Board = np.ndarray[(7, 7), int]
Coord = tuple[int, int]


grid_0 = np.array(
    [
        [0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 6, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0],
    ]
)

view_0 = np.array(
    [
        [5, 7, 0, 0, 0, 5, 7],
        [5, 7, 0, 0, 0, 3, 6],
        [7, 4, 0, 0, 0, 7, 6],
        [5, 4, 0, 0, 0, 7, 5],
    ]
)


grid_1 = np.array(
    [
        [0, 2, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
)

view_1 = np.array(
    [
        [0, 0, 5, 6, 0, 7, 6],
        [6, 7, 5, 0, 0, 0, 0],
        [6, 6, 4, 0, 0, 0, 0],
        [0, 0, 5, 6, 0, 6, 7],
    ]
)


grid_2 = np.array(
    [
        [0, 0, 0, 0, 4, 0, 0],
        [0, 6, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 6],
        [0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 6, 0],
        [0, 0, 4, 0, 0, 0, 0],
    ]
)

view_2 = np.array(
    [
        [0, 0, 0, 7, 0, 0, 0],
        [0, 7, 0, 3, 0, 0, 5],
        [0, 0, 0, 5, 0, 0, 0],
        [7, 0, 0, 5, 0, 7, 0],
    ]
)


grid_3 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0],
    ]
)

view_3 = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7],
        [0, 6, 0, 5, 0, 4, 0],
        [0, 6, 0, 4, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
)


grids = [grid_0, grid_1, grid_2, grid_3]
views = [view_0, view_1, view_2, view_3]


# Utility functions
# ----------------------------------------------------------------------
def connected(xm: Board) -> bool:
    """True iff the board is connected."""
    return label(xm)[1] <= 1


def evaluate_vars(m: ModelRef, vars: np.ndarray) -> np.ndarray:
    """Evaluate variables in a z3 model."""
    return np.vectorize(lambda x: m.evaluate(x).as_long())(vars)


def rowcol_coords(side_idx: int, rowcol_idx: int) -> list[Coord]:
    """Return the coordinates of the row or column."""
    is_row = not (side_idx % 2)  # True if rowcol_idx is the idx of a row
    is_inc = (side_idx + 1) % 4 < 2  # True if the other idx is increasing
    idx_range = list(range(7)) if is_inc else list(reversed(range(7)))
    coord = lambda idx: (rowcol_idx, idx) if is_row else (idx, rowcol_idx)
    coords = [coord(idx) for idx in idx_range]
    return coords


# Variables, solver, constraints
# ----------------------------------------------------------------------


def solution(grid: Board, view: np.ndarray) -> Board | None:
    X = np.array(IntVector("x", 7**2)).reshape(7, 7)
    s = Solver()

    # Given numbers
    s += [x == v for x, v in zip(X.flat, grid.flat, strict=False) if v]

    # Grid contains one 1, two 2’s, etc., up to seven 7’s
    s += [PbEq([(x == k, 1) for x in X.flat], k) for k in range(1, 8)]

    # Each row and column must contain exactly 4 numbers which sum to 20
    for i in range(7):
        s += sum(X[i, :]) == 20
        s += sum(X[:, i]) == 20
        s += PbEq([(x != 0, 1) for x in X[i, :]], 4)
        s += PbEq([(x != 0, 1) for x in X[:, i]], 4)

    # Every 2-by-2 subsquare must contain at least one empty cell
    for i, j in product(range(7 - 1), repeat=2):
        dirs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        s += Or([X[i + di, j + dj] == 0 for di, dj in dirs])

    # Numbers outside of a grid indicate the first number that is “viewable”
    def first_num_seen(coords: list[Coord], num: int, k: int) -> BoolRef:
        zeros_before_k = And([X[i, j] == 0 for i, j in coords[:k]])
        first_num = X[coords[k][0], coords[k][1]] == num
        return And(zeros_before_k, first_num)

    for side_idx, side_nums in enumerate(view):
        for rowcol_idx, num in enumerate(side_nums):
            if num:
                coords = rowcol_coords(side_idx, rowcol_idx)
                s += Or([first_num_seen(coords, num, k) for k in range(4)])

    # Wrap solver in a function which also checks connectivity
    while s.check() == sat:
        m = s.model()
        xm = evaluate_vars(m, X)
        if connected(xm):
            return xm
        else:
            s += Not(And([x == y for x, y in zip(X.flat, xm.flat, strict=False)]))
    return None


# Solve problem
# ----------------------------------------------------------------------

sols = [solution(grid, view) for grid, view in zip(grids, views, strict=False)]
if all(xm is not None for xm in sols):
    ans = np.sum(sum(sols) ** 2)
    print(f"\n{ans = }")
    for i, xm in enumerate(sols):
        print(f"\nxm_{i} = \n", xm)
else:
    print("No solution found.")
# ans = 8520

# xm_0 =
# [[5 4 4 0 0 7 0]
#  [0 0 7 6 3 4 0]
#  [5 6 2 0 7 0 0]
#  [0 3 0 0 6 6 5]
#  [5 7 0 6 0 0 2]
#  [5 0 7 1 0 0 7]
#  [0 0 0 7 4 3 6]]

# xm_1 =
# [[7 2 5 6 0 0 0]
#  [2 0 0 5 7 6 0]
#  [5 7 4 0 0 4 0]
#  [0 0 6 0 4 3 7]
#  [6 4 0 0 3 0 7]
#  [0 7 5 3 0 0 5]
#  [0 0 0 6 6 7 1]]

# xm_2 =
# [[7 2 0 0 4 7 0]
#  [0 6 7 5 2 0 0]
#  [4 5 0 5 0 0 6]
#  [0 0 0 7 7 1 5]
#  [6 7 3 0 0 0 4]
#  [3 0 6 0 0 6 5]
#  [0 0 4 3 7 6 0]]

# xm_3 =
# [[0 0 0 1 5 7 7]
#  [0 2 5 7 0 0 6]
#  [3 7 0 7 0 0 3]
#  [4 0 0 0 5 7 4]
#  [0 5 6 5 4 0 0]
#  [6 0 6 0 6 2 0]
#  [7 6 3 0 0 4 0]]
