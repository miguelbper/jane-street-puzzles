from collections import Counter
from itertools import product
from math import prod

import cpmpy as cp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cpmpy.expressions.variables import NDVarArray
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
from tqdm import tqdm

ARROWS = ["↑", "↓", "←", "→"]
DIRECTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
N = 20

nums_dict = {
    (1, 11): 4,
    (1, 15): 4,
    (2, 7): 5,
    (3, 12): 7,
    (3, 14): 5,
    (4, 10): 4,
    (4, 13): 7,
    (4, 17): 4,
    (5, 6): 4,
    (5, 8): 7,
    (6, 7): 9,
    (7, 17): 6,
    (8, 2): 7,
    (8, 11): 5,
    (9, 14): 5,
    (10, 5): 4,
    (10, 7): 7,
    (10, 18): 3,
    (13, 3): 5,
    (13, 6): 6,
    (13, 9): 2,
    (15, 6): 5,
    (17, 12): 5,
    (17, 13): 5,
    (18, 8): 4,
}

arrows_dict = {
    (0, 1): "→",
    (0, 8): "←↓",
    (0, 12): "↓",
    (0, 15): "↓",
    (0, 19): "←↓",
    (1, 4): "→",
    (1, 13): "←↓→",
    (2, 6): "→",
    (2, 9): "←↓→",
    (2, 18): "←↓↑",
    (3, 1): "↓",
    (3, 3): "↓→",
    (4, 0): "→",
    (4, 5): "↓→",
    (4, 15): "↑←→",
    (5, 11): "↓→",
    (6, 3): "↓",
    (6, 10): "←→",
    (6, 13): "↑↓←",
    (6, 16): "→",
    (6, 19): "←↓",
    (7, 1): "↓→",
    (7, 14): "←↓",
    (8, 5): "↑↓→",
    (8, 9): "↑↓←→",
    (9, 1): "↑→",
    (9, 12): "↑←→",
    (9, 16): "↓→",
    (9, 18): "↑↓←",
    (10, 1): "↑",
    (10, 3): "↑↓",
    (11, 8): "↑←→",
    (11, 10): "←→",
    (11, 14): "↑←→",
    (11, 17): "←↑",
    (12, 2): "↑→",
    (12, 5): "↑↓←→",
    (12, 18): "↑",
    (13, 0): "→",
    (13, 12): "←↑",
    (13, 16): "↑",
    (14, 8): "↑↓←",
    (14, 11): "↑↓",
    (14, 13): "↓",
    (15, 2): "↑→",
    (15, 4): "↑",
    (15, 9): "←↓",
    (15, 14): "←↓",
    (15, 19): "←",
    (16, 5): "↑",
    (16, 7): "↑→",
    (16, 16): "←",
    (16, 18): "←↑",
    (17, 1): "→",
    (17, 10): "↑↓←",
    (18, 4): "↑→",
    (18, 6): "→",
    (18, 14): "←↑",
    (19, 0): "→",
    (19, 4): "↑→",
    (19, 7): "→",
    (19, 11): "←",
    (19, 18): "←↑",
}

nums = np.zeros((N, N), dtype=np.int32)
for (i, j), n in nums_dict.items():
    nums[i, j] = n

arrows = np.zeros((N, N, 4), dtype=np.int32)
for (i, j), arrows_str in arrows_dict.items():
    for arrow in arrows_str:
        arrow_idx = ARROWS.index(arrow)
        arrows[i, j, arrow_idx] = 1


# Constraints
# ------------------------------------------------------------------------------


def add_connected(m: cp.Model, X: NDVarArray, i0: int, j0: int, num_steps: int) -> None:
    """Add connectivity constraint ensuring all True cells in X are reachable
    from (i0, j0).

    Args:
        m: CPMpy constraint model
        X: Boolean variable array representing the grid
        i0: Starting row coordinate
        j0: Starting column coordinate
        num_steps: Maximum number of steps allowed for reachability
    """
    # C[s, i, j] = "starting from (i0, j0), can reach (i, j) in s steps"
    n = X.shape[0]
    C = cp.boolvar(shape=(num_steps, n, n))

    def neighbours(i: int, j: int) -> list[tuple[int, int]]:
        """Return the coordinates of the neighbours of (i, j)."""
        direc = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        neigh = [(i + di, j + dj) for di, dj in direc]
        return [(x, y) for x, y in neigh if 0 <= x < n and 0 <= y < n]

    m += [~c for (i, j), c in np.ndenumerate(C[0]) if (i, j) != (i0, j0)]
    m += X.implies(C[num_steps - 1])
    m += [C[s].implies(X) for s in range(num_steps)]
    for s, i, j in product(range(num_steps - 1), range(n), range(n)):
        m += C[s + 1, i, j].implies(cp.any([C[s, a, b] for a, b in neighbours(i, j)]))


def add_arrows_not_in_box(m: cp.Model, X: NDVarArray) -> None:
    """Add constraint that cells with arrows cannot be inside the box.

    Args:
        m: CPMpy constraint model
        X: Boolean variable array representing the grid
    """
    exists_arrow = np.any(arrows, axis=2)
    m += [~X[i, j] for (i, j), arrow in np.ndenumerate(exists_arrow) if arrow]


def add_nums_in_box(m: cp.Model, X: NDVarArray) -> None:
    """Add constraint that all cells with numbers must be inside the box.

    Args:
        m: CPMpy constraint model
        X: Boolean variable array representing the grid
    """
    m += [X[i, j] for (i, j), num in np.ndenumerate(nums) if num]


def add_nums_king_move(m: cp.Model, X: NDVarArray) -> None:
    """Add constraint that each number equals the count of box cells in its
    king-move neighborhood.

    The king-move neighborhood includes the cell itself and all 8 adjacent cells (like a king in chess).

    Args:
        m: CPMpy constraint model
        X: Boolean variable array representing the grid
    """
    for (i, j), num in np.ndenumerate(nums):
        if num:
            i0 = max(0, i - 1)
            j0 = max(0, j - 1)
            i1 = min(N, i + 2)
            j1 = min(N, j + 2)
            m += cp.sum(X[i0:i1, j0:j1]) == num


def add_first_seen(m: cp.Model, X: NDVarArray) -> None:
    """Add constraint for all arrow cells that they must see a box cell in
    their arrow directions.

    Args:
        m: CPMpy constraint model
        X: Boolean variable array representing the grid
    """
    exists_arrow = np.any(arrows, axis=2)
    for (i, j), exists in np.ndenumerate(exists_arrow):
        if exists:
            arrow = arrows[i, j]
            add_first_seen_by_cell(m, X, i, j, arrow)


def max_steps(i: int, j: int, di: int, dj: int) -> int:
    """Calculate maximum steps possible from (i, j) in direction (di, dj)
    before hitting boundary.

    Args:
        i: Row coordinate
        j: Column coordinate
        di: Row direction (-1, 0, or 1)
        dj: Column direction (-1, 0, or 1)

    Returns:
        Maximum number of steps possible in the given direction
    """
    if di == 1:
        return (N - 1) - i
    elif di == -1:
        return i
    elif dj == 1:
        return (N - 1) - j
    elif dj == -1:
        return j
    return float("inf")


def add_first_seen_by_cell(m: cp.Model, X: NDVarArray, i: int, j: int, arrow: NDArray[np.int32]) -> None:
    """Add constraint that cell (i, j) with arrows must see a box cell at the
    same distance in all arrow directions.

    Args:
        m: CPMpy constraint model
        X: Boolean variable array representing the grid
        i: Row coordinate of the arrow cell
        j: Column coordinate of the arrow cell
        arrow: Array indicating which arrow directions are present
    """
    # Compute maximal k. k = 1,...,max_k.
    max_k = min(max_steps(i, j, di, dj) for (di, dj), a in zip(DIRECTIONS, arrow) if a)
    cell = np.array([i, j])

    # Create k variables & their meaning
    # Z[k] = True iff X=1 is seen at distance k from (i, j), k = 1,...,max_k.
    Z = cp.boolvar(shape=(max_k + 1,), name=f"Z_{i}_{j}")
    m += ~Z[0]

    for k in range(1, max_k + 1):
        cond = []
        for direction, a in zip(DIRECTIONS, arrow):
            if a:
                cond.extend([~X[tuple(cell + step * direction)] for step in range(1, k)])
                cond.append(X[tuple(cell + k * direction)])
            else:
                di, dj = direction
                max_step = max_steps(i, j, di, dj)
                cond.extend([~X[tuple(cell + step * direction)] for step in range(1, min(k, max_step) + 1)])

        m += Z[k] == cp.all(cond)

    # Add Or over those k variables
    m += cp.any(Z)


def solve() -> NDArray[np.int32]:
    """Solve the puzzle and return a grid showing which cells are provably
    in/out of the box.

    Returns:
        NxN array where 0 = provably outside box, 1 = provably inside box, 2 = uncertain
    """
    m = cp.Model()
    X = cp.boolvar(shape=(N, N), name="X")  # X[i, j] = True iff cell (i, j) is part of the box

    # Add constraints
    # --------------------------------------------------------------------------
    Y = cp.boolvar(shape=(N + 2, N + 2), name="Y")  # Shifted and negated X, for connectivity of zeros
    m += Y[1:-1, 1:-1] == ~X
    m += Y[0, :]
    m += Y[-1, :]
    m += Y[:, 0]
    m += Y[:, -1]

    add_connected(m, X, 8, 11, 2 * N)
    add_connected(m, Y, 0, 0, 3 * N)

    add_arrows_not_in_box(m, X)
    add_nums_in_box(m, X)
    add_nums_king_move(m, X)
    add_first_seen(m, X)

    # fast_mode = assume facts that you would only know after solving the puzzle
    # enabled to speed up the script
    # if fast_mode = False, will try to solve from scratch
    fast_mode = True
    if fast_mode:
        m += X[1, 16]
        m += ~X[2, 16]
        m += X[0, 17]
        m += X[3, 19]
        m += X[4, 19]
        m += ~X[17, 15]
        m += ~X[0, 5]
        m += X[0, 6]

    # Find one solution
    # --------------------------------------------------------------------------
    m.solve()
    xm_raw = X.value()
    xm = np.full_like(xm_raw, fill_value=2, dtype=int)
    xm[xm_raw == True] = 1  # noqa: E712
    xm[xm_raw == False] = 0  # noqa: E712

    # Prove properties: find cells that are equal for all solutions
    # --------------------------------------------------------------------------
    if fast_mode:
        A = xm.copy()
    else:
        A = 2 * np.ones((N, N), dtype=np.int32)
        for (i, j), x in tqdm(np.ndenumerate(xm), total=N**2, desc="Proving properties"):
            # Try proving that X[i, j] == x
            m_copy = m.copy()
            m_copy += X[i, j] != x
            proof = not m_copy.solve()
            if proof:
                A[i, j] = x
    return A


def plot(xm: NDArray[np.int32]) -> None:
    """Plot an NxN grid with numbers and arrows."""
    # Create white grid with black lines using seaborn
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create heatmap with three colors: light gray (0), light yellow (1), white (2)
    cmap = ListedColormap(["#D3D3D3", "#FFFACD", "white"])  # light gray, light yellow, white
    sns.heatmap(
        xm,
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        vmin=0,
        vmax=2,
    )

    # Draw numbers
    for i in range(N):
        for j in range(N):
            if nums[i, j] != 0:
                # Center of cell in heatmap coordinates
                x = j + 0.5
                y = i + 0.5
                ax.text(x, y, str(nums[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw arrows
    # Draw arrows from the arrows array (4 basic directions)
    arrow_length = 0.5  # Length of arrow in cell units

    for i in range(N):
        for j in range(N):
            # Center of cell in heatmap coordinates
            x_center = j + 0.5
            y_center = i + 0.5

            # Determine which arrows to draw
            arrow_directions = []
            for arrow_idx, arrow_char in enumerate(ARROWS):
                if arrows[i, j, arrow_idx] == 1:
                    arrow_directions.append(arrow_char)

            # Also check arrows_dict for any additional arrow types
            if (i, j) in arrows_dict:
                arrow_text = arrows_dict[(i, j)]
                # Parse the arrow string to get individual directions
                arrow_directions = list(arrow_text)

            # Draw each arrow emanating from center
            for arrow_char in arrow_directions:
                dx, dy = 0, 0

                if arrow_char == "→":
                    dx, dy = arrow_length, 0
                elif arrow_char == "←":
                    dx, dy = -arrow_length, 0
                elif arrow_char == "↑":
                    dx, dy = 0, -arrow_length
                elif arrow_char == "↓":
                    dx, dy = 0, arrow_length

                if dx != 0 or dy != 0:
                    # Draw arrow from center
                    arrow = mpatches.FancyArrowPatch(
                        (x_center, y_center),
                        (x_center + dx, y_center + dy),
                        arrowstyle="-|>",
                        mutation_scale=10,
                        linewidth=1,
                        color="black",
                        zorder=10,
                    )
                    ax.add_patch(arrow)

    plt.tight_layout()
    plt.show()


A = solve()
plot(A)
print("A = \n", A, sep="")

area_lb = np.sum(A == 1)
area_ub = np.sum(A >= 1)
print(f"{area_lb = }")
print(f"{area_ub = }")

for x, y, z in product(range(1, N), range(1, N), range(1, N)):
    if not (x <= y <= z):
        continue
    area = 2 * (x * y + y * z + z * x)
    if not (area_lb <= area <= area_ub):
        continue
    print(f"{(x, y, z) = }, {area = }")
"""
area_lb = np.int64(131)
area_ub = np.int64(139)

(x, y, z) = (1, 3, 16), area = 134
(x, y, z) = (1, 4, 13), area = 138
(x, y, z) = (1, 6,  9), area = 138
(x, y, z) = (2, 2, 16), area = 136
(x, y, z) = (2, 3, 12), area = 132
(x, y, z) = (2, 4, 10), area = 136
(x, y, z) = (2, 5,  8), area = 132
(x, y, z) = (2, 6,  7), area = 136 <- This is the correct shape
(x, y, z) = (3, 3, 10), area = 138
(x, y, z) = (3, 4,  8), area = 136

Method I used for the last part of the puzzle:
- Print out the grid and cut out the 0 cells.
- Notice that the left portion of the image perfectly fits the "hole" in the center region.
- Manually fold the paper and use tape to make the figure stay in its shape.
- Arrive at a shape of 2x6x7
"""
faces = [
    [4, 5, 5, 5, 4, 7, 4, 4, 7, 3, 9],
    [6, 5],
    [7, 5, 4, 6, 2, 4],
    [5],
    [7, 4],
    [5, 5, 7],
]

c1 = Counter(faces[0] + faces[1] + faces[2] + faces[3] + faces[4] + faces[5])
c2 = Counter([n for (i, j), n in np.ndenumerate(nums) if n])
assert c1 == c2

ans = prod(sum(face) for face in faces)
print(f"{ans = }")
