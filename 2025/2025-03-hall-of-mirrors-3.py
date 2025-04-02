from math import prod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from codetiming import Timer
from numpy.typing import NDArray
from sympy import divisors  # type: ignore
from z3 import And, IntVector, ModelRef, Not, PbEq, Solver, sat  # type: ignore

# Ignore errors thrown by matplotlib
# pyright: reportUnknownMemberType=false

# fmt: off
nums: NDArray[np.int32] = np.array([
    [   0, 0, 0, 27,    0,  0, 0,  12, 225, 0],  # left
    [   1, 0, 0,  9, 3087, 48, 0, 112,   0, 0],  # top
    [   0, 0, 0, 16,    0,  0, 0,  27,   4, 0],  # right
    [2025, 0, 0, 12,   64,  5, 0, 405,   0, 0],  # bottom
], dtype=np.int32)
# fmt: on

Cell = NDArray[np.int32]  # (2,) [i, j]
Direction = NDArray[np.int32]  # (2,) [di, dj]
Product = int
Mirror = int
Path = list[tuple[Cell, Mirror]]
Initial = bool
State = tuple[Cell, Direction, Product, Path, Initial]

DIRECTIONS: NDArray[np.int32] = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype=np.int32)

# Get a list of colors for the lasers. Shuffle to get different colors in similar places of the grid
np.random.seed(0)
COLORS = np.random.permutation(sns.color_palette("husl", n_colors=40))


def compute_cell(side: int, k: int) -> Cell:
    """Convert side and position to cell coordinates.

    Args:
        side: Side of the grid (0=left, 1=top, 2=right, 3=bottom)
        k: Position along the side (0-9)

    Returns:
        Cell coordinates [i, j] including outside grid positions
    """
    a: int = k if side in [0, 3] else 9 - k
    b: int = -1 if side in [0, 1] else 10
    i, j = (a, b) if side % 2 == 0 else (b, a)
    return np.array([i, j], dtype=np.int32)


def compute_side(cell: Cell) -> tuple[int, int]:
    """Convert cell coordinates to side and position.

    Args:
        cell: Cell coordinates [i, j]

    Returns:
        Tuple of (side, k) where side is 0-3 and k is position 0-9
    """
    i, j = cell
    if j == -1:  # left
        side, k = 0, i
    elif i == -1:  # top
        side, k = 1, 9 - j
    elif j == 10:  # right
        side, k = 2, 9 - i
    else:  # bottom (i == 10)
        side, k = 3, j
    return side, k


def distance(cell: Cell) -> int:
    """Compute distance from cell to grid boundary.

    Args:
        cell: Cell coordinates [i, j]

    Returns:
        Distance to boundary:
        - Negative inside grid
        - Zero on boundary
        - Positive outside grid
    """
    return int(np.max(np.floor(np.abs(cell - 4.5)))) - 4


def reject(state: State) -> bool:
    """Check if a state should be rejected.

    Args:
        state: Current state (cell, direction, product, path, initial)

    Returns:
        True if state should be rejected, False otherwise
    """
    cell, _, remaining_product, _, initial = state
    dist: int = distance(cell)
    conditions: list[bool] = [
        dist > 1,
        dist == 1 and remaining_product != 1 and not initial,
        dist < 0 and remaining_product == 1,
    ]
    return any(conditions)


def accept(state: State) -> bool:
    """Check if a state is a solution.

    Args:
        state: Current state (cell, direction, product, path, initial)

    Returns:
        True if state is a valid solution, False otherwise
    """
    cell, _, remaining_product, _, initial = state
    return distance(cell) == 1 and remaining_product == 1 and not initial


def branch(state: State) -> list[State]:
    """Generate all possible next states from current state.

    Args:
        state: Current state (cell, direction, product, path, initial)

    Returns:
        List of possible next states
    """
    cell, direction, remaining_product, path, initial = state
    new_states: list[State] = []
    divs: list[int] = list(divisors(remaining_product))
    mirrors: list[Mirror] = [0] if initial else [1, 2]

    for mirror in mirrors:
        sign: int = -1 if mirror == 1 else 1
        new_direction: Direction = direction if mirror == 0 else sign * np.flip(direction)
        for divisor in divs:
            new_cell: Cell = cell + new_direction * divisor
            if divisor == 1 and distance(cell) <= 0 and distance(new_cell) <= 0:
                continue
            new_product: Product = remaining_product // divisor
            path_continuation: Path = [(cell + k * new_direction, 0 if k else mirror) for k in range(divisor)]
            new_path: Path = path + [(c, x) for (c, x) in path_continuation if distance(c) <= 0]
            new_state: State = (new_cell, new_direction, new_product, new_path, False)
            new_states.append(new_state)

    return new_states


def paths(initial_cell: Cell, initial_direction: Direction, initial_product: Product) -> list[State]:
    """Find all valid paths from initial state.

    Args:
        initial_cell: Starting cell coordinates
        initial_direction: Starting direction
        initial_product: Target product to achieve

    Returns:
        List of valid final states
    """
    solutions: list[State] = []
    initial_state: State = (initial_cell, initial_direction, initial_product, [], True)
    stack: list[State] = [initial_state]

    while stack:
        state: State = stack.pop()
        if reject(state):
            continue
        if accept(state):
            solutions.append(state)
        else:
            stack.extend(branch(state))

    return solutions


def laser(xm: NDArray[np.int32], side: int, k: int) -> list[Cell]:
    """Trace laser path through grid of mirrors.

    Args:
        xm: Grid of mirrors (0=empty, 1=/, 2=\\)
        side: Starting side (0-3)
        k: Starting position on side (0-9)

    Returns:
        List of cells the laser passes through
    """
    cell: Cell = compute_cell(side, k)
    direction: Direction = DIRECTIONS[side]
    squares: list[Cell] = [cell]

    while True:
        cell = cell + direction
        if distance(cell) > 0:
            squares.append(cell)
            break
        elif (x := xm[*cell]) > 0:
            squares.append(cell)
            sign: int = -1 if x == 1 else 1
            direction: NDArray[np.int32] = sign * np.flip(direction)

    return squares


def product(path: list[Cell]) -> int:
    """Compute product of Manhattan distances between consecutive cells.

    Args:
        path: List of cell coordinates

    Returns:
        Product of distances
    """
    return prod(np.sum(np.abs(s - t)) for s, t in zip(path, path[1:]))


def answer(xm: NDArray[np.int32]) -> int:
    """Compute final answer from grid of mirrors.

    Args:
        xm: Grid of mirrors (0=empty, 1=/, 2=\\)

    Returns:
        Product of sums of products for each side
    """
    laser_arr: NDArray[np.object_] = np.empty((4, 10), dtype=np.object_)
    for side, k in np.ndindex(laser_arr.shape):
        laser_arr[side, k] = laser(xm, side, k)
    products: NDArray[np.int32] = np.vectorize(product)(laser_arr)
    nonzero_products: NDArray[np.int32] = np.where(nums, 0, products)
    return np.prod(np.sum(nonzero_products, axis=1), axis=0)


def plot(xm: NDArray[np.int32]) -> None:
    """Plot grid with mirrors, lasers, and products.

    Args:
        xm: Grid of mirrors (0=empty, 1=/, 2=\\)
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(np.zeros((10, 10)), cmap=["white"], cbar=False, xticklabels=False, yticklabels=False)
    plt.gca().set_aspect("equal")
    plt.xlim(-2, 12)
    plt.ylim(12, -2)

    # Draw grid lines
    for i in range(11):
        plt.plot([0, 10], [i, i], color="black", linewidth=0.5)
        plt.plot([i, i], [0, 10], color="black", linewidth=0.5)

    # Draw bold outer grid border
    plt.plot([0, 10], [0, 0], color="black", linewidth=2)
    plt.plot([0, 10], [10, 10], color="black", linewidth=2)
    plt.plot([0, 0], [0, 10], color="black", linewidth=2)
    plt.plot([10, 10], [0, 10], color="black", linewidth=2)

    # Draw exterior black dots
    for j in range(10):
        plt.plot(j + 0.5, -0.5, "ko")
        plt.plot(j + 0.5, 10.5, "ko")
    for i in range(10):
        plt.plot(-0.5, i + 0.5, "ko")
        plt.plot(10.5, i + 0.5, "ko")

    # Draw exterior numbers
    for side, k in np.argwhere(nums):
        num: int = int(nums[side, k])
        i, j = compute_cell(side, k) - DIRECTIONS[side]
        plt.text(j + 0.5, i + 0.5, str(num), ha="center", va="center", fontsize=12)

    # Draw mirrors
    for (i, j), x in np.ndenumerate(xm):
        match int(x):
            case 1:
                plt.plot([j, j + 1], [i + 1, i], "red", linewidth=2)
            case 2:
                plt.plot([j, j + 1], [i, i + 1], "red", linewidth=2)
            case _:
                pass

    products: NDArray[np.int32] = np.zeros((4, 10), dtype=np.int32)
    color_idx: int = 0

    # Draw lasers and their products
    for side, k in np.ndindex(products.shape):
        if products[side, k] > 0:
            continue

        num: int = nums[side, k]
        cells: list[Cell] = laser(xm, side, k)
        prd: int = product(cells)
        color = COLORS[color_idx]
        color_idx += 1

        # Draw product on both ends
        for cell in [cells[0], cells[-1]]:
            side_, k_ = compute_side(cell)
            products[side_, k_] = prd
            if not nums[side_, k_]:
                i, j = cell - DIRECTIONS[side_]
                plt.text(j + 0.5, i + 0.5, str(prd), ha="center", va="center", color="red", fontsize=12)

        # Draw laser
        for s, t in zip(cells, cells[1:]):
            plt.plot([s[1] + 0.5, t[1] + 0.5], [s[0] + 0.5, t[0] + 0.5], color=color, linewidth=1.5)

    plt.show()


with Timer(initial_text="Solving puzzle..."):
    s = Solver()
    X = np.array(IntVector("x", 10**2)).reshape((10, 10))
    s += [And(x >= 0, x <= 2) for x in X.flat]
    s += [Not(And(x > 0, y > 0)) for x, y in zip(X[:, 1:].flat, X[:, :-1].flat)]
    s += [Not(And(x > 0, y > 0)) for x, y in zip(X[1:].flat, X[:-1].flat)]

    for side, k in np.argwhere(nums):
        num: int = int(nums[side, k])
        initial_cell: Cell = compute_cell(side, k)
        initial_direction: Direction = DIRECTIONS[side]
        initial_product: Product = num
        states: list[State] = paths(initial_cell, initial_direction, initial_product)
        paths_: list[Path] = [path for _, _, _, path, _ in states]
        s += PbEq([(And([X[*cell] == x for cell, x in path]), 1) for path in paths_], 1)

    check_ = s.check()

if check_ == sat:
    m: ModelRef = s.model()
    xm: NDArray[np.int32] = np.vectorize(lambda x: m.evaluate(x).as_long())(X)  # type: ignore
    ans: int = answer(xm)
    print(f"Answer: {ans}")
    plot(xm)
else:
    print("No solution found.")
