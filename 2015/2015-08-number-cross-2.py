from math import prod

import numpy as np
from codetiming import Timer
from z3 import And, ArithRef, IntVector, ModelRef, Or, Tactic, sat

# Grid and clues
# ----------------------------------------------------------------------
B = 0
W = -1
# fmt: off
grid = np.array([
    [1 , 2 , 3 , 4 , B , 5 , 6 , 7 , 8 ],
    [9 , W , W , W , B , 10, W , W , W ],
    [B , 11, W , W , 12, B , 13, W , W ],
    [14, W , W , W , W , 15, W , B , B ],
    [16, W , B , 17, W , W , B , 18, 19],
    [B , B , 20, W , W , W , 21, W , W ],
    [22, 23, W , B , 24, W , W , W , B ],
    [25, W , W , 26, B , 27, W , W , 28],
    [29, W , W , W , B , 30, W , W , W ],
])
# fmt: on
n = len(grid)


clues_across = {
    1: 9,
    5: 35,
    9: 10,
    10: 30,
    11: 7,
    13: 10,
    14: 42,
    16: 21,
    17: 25,
    18: 15,
    20: 120,
    22: 25,
    24: 35,
    25: 21,
    27: 9,
    29: 5,
    30: 8,
}

clues_down = {
    1: 45,
    2: 20,
    3: 48,
    4: 72,
    5: 18,
    6: 24,
    7: 27,
    8: 26,
    12: 12,
    14: 18,
    15: 32,
    18: 45,
    19: 20,
    20: 30,
    21: 12,
    22: 70,
    23: 12,
    26: 2,
    28: 36,
}

clues = [clues_down, clues_across]


# Helper functions
# ----------------------------------------------------------------------
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


# Variables, solver & constraints
# ----------------------------------------------------------------------
s = Tactic("qfnia").solver()  # Quantifier-Free Non-linear Integer Arithmetic
X = np.array(IntVector("x", n**2)).reshape((n, n))

# Ranges for each variable + black cells
s += [And(x > 0, x < 10) for (i, j), x in np.ndenumerate(X) if grid[i, j] != B]
s += [x == B for (i, j), x in np.ndenumerate(X) if grid[i, j] == B]


def variables(i: int, j: int, dim: int) -> list[ArithRef]:
    """Returns a list of variables in a given direction."""
    ans = []
    i_, j_ = i, j
    di = int(dim == 0)
    dj = int(dim == 1)
    in_bounds = lambda i_, j_: 0 <= i_ < n and 0 <= j_ < n  # noqa: E731
    blocked = lambda i_, j_: grid[i_, j_] == B  # noqa: E731
    while in_bounds(i_, j_) and not blocked(i_, j_):
        ans.append(X[i_, j_])
        i_ += di
        j_ += dj
    return ans


# Loop over grid cells containing a clue
for (i, j), clue in np.ndenumerate(grid):
    if clue in [B, W]:
        continue

    # Loop over directions (across, down) where clue is present
    for dim in range(2):
        if clue not in clues[dim]:
            continue

        # Get the value and add the constraint
        value = clues[dim][clue]
        equal_prd = prod(variables(i, j, dim)) == value
        equal_sum = sum(variables(i, j, dim)) == value
        s += Or(equal_prd, equal_sum)


# Solve
# ----------------------------------------------------------------------

with Timer(initial_text="Checking z3 solver"):
    check = s.check()

if check == sat:
    m = s.model()
    xm = evaluate_vars(m, X)
    xm_str = np.vectorize(lambda x: "." if x == B else str(x))(xm)
    print(f"answer = {np.sum(xm)}")
    print_arr(xm_str, "xm")
# Checking z3 solver
# Elapsed time: 4.3804 seconds
# answer = 276
# xm = [9 1 1 1 . 9 8 9 9]
#      [5 1 2 1 . 9 9 3 9]
#      [. 2 3 1 1 . 1 1 8]
#      [6 9 8 8 1 4 6 . .]
#      [3 7 . 9 8 8 . 3 5]
#      [. . 5 1 1 1 2 3 4]
#      [5 5 1 . 1 1 7 5 .]
#      [7 6 6 2 . 1 1 1 9]
#      [2 1 1 1 . 1 2 1 4]
