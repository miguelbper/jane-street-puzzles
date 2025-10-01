from itertools import product

import cpmpy as cp
import numpy as np
from codetiming import Timer
from cpmpy.expressions.variables import NDVarArray
from numpy.typing import NDArray
from scipy.ndimage import label, sum_labels

N = 9
NUM_DIGITS = sum(range(1, N + 1))  # 45
NUM_PENTOS = 12


# fmt: off
CLUES = np.array([
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 0, 0],
])
# fmt: on

# Pentominos
# --------------------------------------------------------------------------

pentomino_1 = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, 0],
])  # fmt: skip

pentomino_2 = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 1],
])  # fmt: skip

pentomino_3 = np.array([
    [1, 1],
    [1, 1],
    [0, 1],
])  # fmt: skip

pentomino_4 = np.array([  # N
    [0, 1],
    [0, 1],
    [1, 1],
    [1, 0],
])  # fmt: skip

pentomino_5 = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0],
])  # fmt: skip

pentomino_6 = np.array([  # U
    [1, 0, 1],
    [1, 1, 1],
])  # fmt: skip

pentomino_7 = np.array([  # V
    [0, 0, 1],
    [0, 0, 1],
    [1, 1, 1],
])  # fmt: skip

pentomino_8 = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
])  # fmt: skip

pentomino_9 = np.array([  # X
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
])  # fmt: skip

pentomino_10 = np.array([
    [0, 1],
    [0, 1],
    [1, 1],
    [0, 1],
])  # fmt: skip

pentomino_11 = np.array([  # Z
    [0, 1, 1],
    [0, 1, 0],
    [1, 1, 0],
])  # fmt: skip

pentomino_12 = np.array([[1, 1, 1, 1, 1]])  # I

base_pentominos = [
    pentomino_1,
    pentomino_2,
    pentomino_3,
    pentomino_4,
    pentomino_5,
    pentomino_6,
    pentomino_7,
    pentomino_8,
    pentomino_9,
    pentomino_10,
    pentomino_11,
    pentomino_12,
]

pentominos = [[]]
for pentomino in base_pentominos:
    new_pentominos = []
    for rotation in range(4):
        new_pentominos.append(np.rot90(pentomino, rotation))
        new_pentominos.append(np.rot90(np.fliplr(pentomino), rotation))

    hashables = [tuple(map(tuple, p)) for p in new_pentominos]
    deduplicated = [np.array(p) for p in set(hashables)]
    pentominos.append(deduplicated)


# Constraints
# --------------------------------------------------------------------------


def add_clues(m: cp.Model, X: NDVarArray) -> None:
    """Add constraints for the given clues in the puzzle.

    Args:
        m: The CP model to add constraints to
        X: The variable array representing the puzzle grid
    """
    m += [x == c for c, x in zip(CLUES.flat, X.flat) if c]


def add_two_by_two_unfilled(m: cp.Model, X: NDVarArray) -> None:
    """Add constraints to ensure no 2x2 region is completely filled.

    This constraint ensures that for every 2x2 square in the grid, at least one cell
    must be empty (value 0).

    Args:
        m: The CP model to add constraints to
        X: The variable array representing the puzzle grid
    """
    dirs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, j in product(range(N - 1), repeat=2):
        m += cp.any([X[i + di, j + dj] == 0 for di, dj in dirs])


def add_connectivity(m: cp.Model, X: NDVarArray) -> None:
    """Add constraints to ensure all filled cells are connected.

    This constraint ensures that all non-zero cells in the grid form a connected
    region, starting from the center cell (4, 4). Uses a reachability approach
    where C[s, i, j] represents whether cell (i, j) can be reached in s steps
    from the starting position.

    Args:
        m: The CP model to add constraints to
        X: The variable array representing the puzzle grid
    """
    # C[s, i, j] = "starting from (4, 4), can reach (i, j) in s steps"
    C = cp.boolvar(shape=(NUM_DIGITS, N, N), name="C")

    def neighbours(i: int, j: int) -> list[tuple[int, int]]:
        """Return the coordinates of the neighbours of (i, j)."""
        direc = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        neigh = [(i + di, j + dj) for di, dj in direc]
        return [(x, y) for x, y in neigh if 0 <= x < N and 0 <= y < N]

    m += [~c for (i, j), c in np.ndenumerate(C[0]) if (i, j) != (4, 4)]
    m += (X != 0).implies(C[NUM_DIGITS - 1])
    m += [C[s].implies(X != 0) for s in range(NUM_DIGITS)]
    for s, i, j in product(range(NUM_DIGITS - 1), range(N), range(N)):
        m += C[s + 1, i, j].implies(cp.any([C[s, a, b] for a, b in neighbours(i, j)]))


def add_hooks(m: cp.Model, X: NDVarArray) -> None:
    """Add constraints for the hook regions in the puzzle.

    This constraint defines N hook regions, each containing a unique digit.
    Each hook is either a row or column segment, and all cells in a hook
    must either be empty (0) or contain the hook's assigned digit.

    Args:
        m: The CP model to add constraints to
        X: The variable array representing the puzzle grid
    """
    O = cp.intvar(0, 3, shape=(N,), name="O")
    D = cp.intvar(1, N, shape=(N,), name="D")

    m += cp.AllDifferent(D)

    OI = (O == 0) | (O == 3)
    OJ = (O == 0) | (O == 1)
    I0 = cp.cpm_array([cp.sum(OI[:k]) for k in range(N)])
    J0 = cp.cpm_array([cp.sum(OJ[:k]) for k in range(N)])
    I1 = -cp.cpm_array(range(N)) + I0 + N
    J1 = -cp.cpm_array(range(N)) + J0 + N

    for k, i, j in product(range(N), range(N), range(N)):
        in_row = cp.IfThenElse(OI[k], I0[k] == i, I1[k] - 1 == i) & (J0[k] <= j) & (j < J1[k])
        in_col = cp.IfThenElse(OJ[k], J0[k] == j, J1[k] - 1 == j) & (I0[k] <= i) & (i < I1[k])
        in_hook = in_row | in_col
        m += in_hook.implies((X[i, j] == 0) | (X[i, j] == D[k]))


def add_num_digits(m: cp.Model, X: NDVarArray) -> None:
    """Add constraints to ensure each digit appears exactly its value times.

    This constraint ensures that digit k appears exactly k times in the grid,
    for k = 1, 2, ..., N.

    Args:
        m: The CP model to add constraints to
        X: The variable array representing the puzzle grid
    """
    A = cp.cpm_array(range(1, N + 1))
    m += cp.GlobalCardinalityCount(X, A, A)


def first_seen(
    m: cp.Model,
    X: NDVarArray,
    num: int,
    dim: int,
    idx: int,
    reverse: bool,
) -> None:
    """Add constraints for the "first seen" rule.

    This constraint ensures that when looking along a specific row or column,
    the first non-zero digit seen is the specified number. The constraint
    allows for the possibility that the number appears at any position along
    the specified line.

    Args:
        m: The CP model to add constraints to
        X: The variable array representing the puzzle grid
        num: The digit that should be first seen
        dim: The dimension to look along (0 for rows, 1 for columns)
        idx: The index of the row/column to examine
        reverse: Whether to examine the line in reverse order
    """
    xs = np.take(X, indices=idx, axis=dim)
    xs = xs[::-1] if reverse else xs
    ns = num * np.tri(N, dtype=np.int32).T
    m += cp.any(cp.all(xs[: k + 1] == ns[k, : k + 1]) for k in range(N))


def add_pentominos(
    m: cp.Model,
    X: NDVarArray,
    P: NDVarArray,
) -> None:
    """Add constraints for pentomino placement and visibility rules.

    This constraint ensures that:
    1. Empty cells correspond to no pentomino (P == 0)
    2. Each pentomino appears at most once
    3. Pentominos are placed in valid configurations
    4. The sum of digits in each pentomino is divisible by 5
    5. Specific pentominos must be visible from certain rows

    Args:
        m: The CP model to add constraints to
        X: The variable array representing the puzzle grid
        P: The variable array representing pentomino assignments
    """
    m += (X == 0) == (P == 0)
    m += [cp.sum(p == P) <= 5 for p in range(1, NUM_PENTOS + 1)]

    for p_idx, pentomino_rotations in enumerate(pentominos):
        if not pentomino_rotations:
            continue

        conditions = []

        for pentomino in pentomino_rotations:
            di, dj = pentomino.shape

            for a, b in product(range(N - di + 1), range(N - dj + 1)):
                mask = np.zeros((N, N), dtype=np.bool_)
                mask[a : a + di, b : b + dj] = pentomino

                cond_pent = cp.all(P[mask] == p_idx)
                cond_mult = (cp.sum(X[mask]) % 5) == 0

                match p_idx:
                    case 4:  # N
                        visible = a <= 5 < a + di
                    case 6:  # U
                        visible = a <= 0 < a + di
                    case 7:  # V
                        visible = a <= 8 < a + di
                    case 9:  # X
                        visible = a <= 3 < a + di
                    case 11:  # Z
                        visible = a <= 8 < a + di
                    case 12:  # I
                        visible = a <= 0 < a + di
                    case _:
                        visible = True

                if visible:
                    conditions.append(cond_pent & cond_mult)

        pentomino_used = cp.any(p_idx == P)
        pentomino_cfgs = cp.cpm_array(conditions)
        pentomino_ints = cp.intvar(0, 1, shape=pentomino_cfgs.shape)
        m += pentomino_ints == pentomino_cfgs
        m += pentomino_used.implies(cp.GlobalCardinalityCount(pentomino_ints, cp.cpm_array([1]), cp.cpm_array([1])))


def solve():
    """Solve the hooks puzzle using constraint programming.

    This function sets up the constraint programming model with all the
    necessary constraints and solves the puzzle. It prints the solution
    grid and the answer (product of unfilled region areas) if found.
    """
    m = cp.Model()

    X = cp.intvar(0, N, shape=(N, N), name="X")  # X[i, j] = number in cell (i, j)
    P = cp.intvar(0, NUM_PENTOS, shape=(N, N), name="P")  # P[i, j] = index of pentomino that contains (i, j)

    # Add constraints
    # --------------------------------------------------------------------------
    add_clues(m, X)
    add_two_by_two_unfilled(m, X)
    add_connectivity(m, X)
    add_hooks(m, X)
    add_num_digits(m, X)
    add_pentominos(m, X, P)

    # fmt: off
    first_seen(m, X, num= 3, dim=1, idx=2, reverse= True)
    first_seen(m, X, num= 6, dim=0, idx=3, reverse=False)
    first_seen(m, X, num= 7, dim=1, idx=6, reverse=False)
    first_seen(m, X, num= 2, dim=0, idx=5, reverse= True)
    first_seen(m, P, num=12, dim=0, idx=0, reverse=False)
    first_seen(m, P, num= 6, dim=0, idx=0, reverse= True)
    first_seen(m, P, num= 9, dim=0, idx=3, reverse= True)
    first_seen(m, P, num= 4, dim=0, idx=5, reverse=False)
    first_seen(m, P, num=11, dim=0, idx=8, reverse=False)
    first_seen(m, P, num= 7, dim=0, idx=8, reverse= True)
    # fmt: on

    # Call solver
    # --------------------------------------------------------------------------
    with Timer(initial_text="Solving..."):
        hassol = m.solve()
    print("Status:", m.status())

    if hassol:
        xm = X.value()
        pm = P.value()
        ans = areas(xm)
        print(f"{ans = }")
        print("X = \n", xm, sep="")
        print("P = \n", pm, sep="")
    else:
        print("No solution found.")


def areas(xm: NDArray[np.int32]) -> int:
    """Calculate the product of the areas of the unfilled regions.

    This function identifies all connected regions of empty cells (value 0)
    in the grid and returns the product of their areas.

    Args:
        xm: The solved puzzle grid as a numpy array

    Returns:
        The product of the areas of all unfilled regions
    """
    mat = ~xm.astype(np.bool_)
    labels, k = label(mat)
    area = sum_labels(mat, labels, index=range(1, k + 1))
    return np.prod(area, dtype=np.int32)


if __name__ == "__main__":
    solve()
"""
Solving...
Elapsed time: 20.9612 seconds
Status: ExitStatus.FEASIBLE (13.879702722000001 seconds)
ans = np.int32(1620)

X =
[[0 5 5 5 5 5 7 0 9]
 [0 4 0 4 0 0 7 8 9]
 [0 6 6 6 6 0 0 8 0]
 [0 6 0 0 3 0 7 8 9]
 [4 6 3 0 1 0 0 8 0]
 [4 0 3 2 2 0 0 0 0]
 [7 7 0 0 7 0 7 0 9]
 [0 8 0 0 8 8 8 0 9]
 [0 0 0 0 9 0 9 9 9]]

P =
[[ 0 12 12 12 12 12  6  0  6]
 [ 0  2  0  1  0  0  6  6  6]
 [ 0  2  1  1  1  0  0  9  0]
 [ 0  2  0  0  1  0  9  9  9]
 [ 4  2  2  0  5  0  0  9  0]
 [ 4  0  5  5  5  0  0  0  0]
 [ 4  4  0  0  5  0 11  0  7]
 [ 0  4  0  0 11 11 11  0  7]
 [ 0  0  0  0 11  0  7  7  7]]
"""
