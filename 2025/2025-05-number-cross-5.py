from collections.abc import Callable
from itertools import product
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from codetiming import Timer
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
from sympy import primerange
from tqdm import tqdm
from z3 import And, ArithRef, BoolRef, BoolVector, If, Implies, IntVector, ModelRef, Not, Or, Solver, sat, set_param

set_param("parallel.enable", True)

# Constants & input
# ----------------------------------------------------------------------
N = 11
directions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]], dtype=np.int32)

regions = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 2, 2, 2, 2, 3, 3, 3, 0, 3],
        [1, 2, 2, 1, 2, 4, 3, 3, 3, 3, 3],
        [1, 2, 2, 1, 2, 4, 4, 3, 3, 4, 3],
        [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 3],
        [1, 6, 5, 5, 1, 1, 4, 4, 5, 4, 4],
        [1, 6, 5, 5, 5, 5, 5, 5, 5, 8, 8],
        [6, 6, 6, 6, 5, 6, 5, 8, 8, 8, 8],
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        [6, 6, 7, 7, 7, 7, 7, 7, 6, 6, 6],
    ],
    dtype=np.int32,
)

highlighted = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.bool_,
)


# Utility functions
# ----------------------------------------------------------------------
def neighbours(i: int, j: int, u: int, d: int, l: int, r: int) -> NDArray[np.bool_]:
    """Return a boolean array marking valid neighbors of (i,j) within the given
    up/down/left/right bounds."""
    result = np.zeros((N, N), dtype=np.bool_)
    result[max(0, i - u) : min(N, i + d + 1), j] = True
    result[i, max(0, j - l) : min(N, j + r + 1)] = True
    result[i, j] = False
    return result


def evaluate_vars(m: ModelRef, V: NDArray[Any]) -> NDArray[np.int32]:
    """Convert a Z3 model's variables to a numpy array of integers."""

    def eval_var(v: BoolRef | ArithRef) -> int:
        vm = m.evaluate(v)
        return vm.as_long() if isinstance(vm, ArithRef) else vm.py_value()

    return np.vectorize(eval_var)(V)


def print_array(arr: NDArray[np.bool_ | np.integer], name: str) -> None:
    """Print a named array, converting it to integers for display."""
    print(f"\n{name} = \n", end="")
    print(arr.astype(int))


# cells_around_tile[i, j] = (a, b, k) such that (a, b) + directions[k] = (i, j)
cells_around_tile = np.zeros((N, N, N, N, 4), dtype=np.bool_)
for i, j in product(range(N), repeat=2):
    for k, dk in enumerate(directions):
        a, b = np.array([i, j]) - dk
        if 0 <= a < N and 0 <= b < N:
            cells_around_tile[i, j, a, b, k] = True

# Variables, solver & constraints
# ----------------------------------------------------------------------
# X[i, j]    = digit at position i, j (1-9)
# Y[i, j]    = effective digit at position i, j after displacement (0-9)
# T[i, j]    = whether position i, j has a tile (boolean)
# C[i, j]    = total displacement at i, j
# D[i, j, k] = displacement caused by tile at i, j in direction k (0-9)
X = np.array(IntVector("x", N**2)).reshape((N, N))
Y = np.array(IntVector("y", N**2)).reshape((N, N))
T = np.array(BoolVector("t", N**2)).reshape((N, N))
C = np.array(IntVector("d", N**2)).reshape((N, N))
D = np.array(IntVector("c", N**2 * 4)).reshape((N, N, 4))

s = Solver()

# Variable bounds
s += [And(x >= 1, x <= 9) for x in X.flat]
s += [And(y >= 0, y <= 9) for y in Y.flat]
s += [And(c >= 0, c <= 9) for c in C.flat]
s += [And(d >= 0, d <= 9) for d in D.flat]

# No displacement for border cells in direction off the grid
s += [d == 0 for d in D[:, N - 1, 0]]
s += [d == 0 for d in D[0, :, 1]]
s += [d == 0 for d in D[:, 0, 2]]
s += [d == 0 for d in D[N - 1, :, 3]]

# Relation between C and D
s += [c == sum(ds) for c, ds in zip(C.flat, D.reshape(N**2, 4))]

# Y = X + D if T = 0, Y = 0 if T = 1
s += [If(t, y == 0, y == x + c) for x, y, t, c in zip(X.flat, Y.flat, T.flat, C.flat)]

# Every cell within a region must contain the same digit,
# and orthogonally adjacent cells in different regions must have different digits
for i, j in product(range(N), repeat=2):
    for k, l in np.argwhere(neighbours(i, j, 0, 1, 0, 1)):
        if regions[i, j] == regions[k, l]:
            s += X[i, j] == X[k, l]
        else:
            s += X[i, j] != X[k, l]

# No two tiles are allowed to share a common edge
# Numbers must be at least two digits long
for i, j in product(range(N), repeat=2):
    for k, l in np.argwhere(neighbours(i, j, 0, 1, 0, 2)):
        s += Not(And(T[i, j], T[k, l]))

s += [Not(t) for t in T[:, 1]]
s += [Not(t) for t in T[:, N - 2]]

# When a tile is placed on a cell, it displaces the value (digit) in that cell
for i, j in product(range(N), repeat=2):
    neighbors_sum = sum(D[cells_around_tile[i, j]])
    s += Implies(T[i, j], neighbors_sum == X[i, j])
    s += Implies(Not(T[i, j]), neighbors_sum == 0)

# Some cells have been highlighted: these cells may not contain tiles and may not be altered by any of the increments
s += [Not(t) for t in T[highlighted].flat]
s += [c == 0 for c in C[highlighted].flat]

# Row clue constraints
# ----------------------------------------------------------------------


def number(X: list[ArithRef]) -> ArithRef:
    """Convert a list of Z3 digit variables into a single number variable."""
    return sum(10**i * x for i, x in enumerate(reversed(X)))


def equals(X: list[ArithRef], num: int) -> BoolRef:
    """Create a Z3 constraint that X equals the given number."""
    return And([x == int(d) for x, d in zip(X, str(num))])


def is_number(i: int, l: int, r: int) -> BoolRef:
    """Check if positions [l:r] in row i form a valid number (bounded by black
    tiles or grid edges)."""
    l_boundary = l == 0 or T[i, l - 1]
    r_boundary = r == N or T[i, r]
    interior = And([Not(t) for t in T[i, l:r]])
    return And(l_boundary, r_boundary, interior)


def multiple_of_13(X: list[ArithRef]) -> BoolRef:
    """Create a Z3 constraint that X is divisible by 13."""
    return number(X) % 13 == 0


def multiple_of_32(X: list[ArithRef]) -> BoolRef:
    """Create a Z3 constraint that X is divisible by 32."""
    return number(X) % 32 == 0


def product_of_digits_is_20(X: list[ArithRef]) -> BoolRef:
    """Create a Z3 constraint that the product of digits in X equals 20."""
    valid_digits = And([Or([x == d for d in [1, 2, 4, 5]]) for x in X])
    num_2s = sum([x == 2 for x in X])
    num_4s = sum([x == 4 for x in X])
    num_5s = sum([x == 5 for x in X])
    prod_20_with_4s = And(num_2s == 0, num_4s == 1, num_5s == 1)
    prod_20_with_2s = And(num_2s == 2, num_4s == 0, num_5s == 1)
    return And(valid_digits, Or(prod_20_with_4s, prod_20_with_2s))


def product_of_digits_is_25(X: list[ArithRef]) -> BoolRef:
    """Create a Z3 constraint that the product of digits in X equals 25."""
    valid_digits = And([Or([x == d for d in [1, 5]]) for x in X])
    num_5s = sum([x == 5 for x in X])
    return And(valid_digits, num_5s == 2)


def product_of_digits_is_2025(X: list[ArithRef]) -> BoolRef:
    """Create a Z3 constraint that the product of digits in X equals 2025."""
    valid_digits = And([Or([x == d for d in [1, 3, 5, 9]]) for x in X])
    num_3s = sum([x == 3 for x in X])
    num_5s = sum([x == 5 for x in X])
    num_9s = sum([x == 9 for x in X])
    prod_2025_with_0_nines = And(num_3s == 4, num_5s == 2, num_9s == 0)
    prod_2025_with_1_nines = And(num_3s == 2, num_5s == 2, num_9s == 1)
    prod_2025_with_2_nines = And(num_3s == 0, num_5s == 2, num_9s == 2)
    return And(valid_digits, Or(prod_2025_with_0_nines, prod_2025_with_1_nines, prod_2025_with_2_nines))


def odd_palindrome(X: list[ArithRef]) -> BoolRef:
    """Create a Z3 constraint that X is both odd and a palindrome."""
    odd = Or([X[-1] == d for d in [1, 3, 5, 7, 9]])
    iterator = list(zip(X, reversed(X)))[: len(X) // 2]
    palindrome = And([x == y for x, y in iterator])
    return And(odd, palindrome)


def clue_from_lists(lists: list[list[int]]) -> Callable[[list[ArithRef]], BoolRef]:
    """Create a function that checks if X matches any number in the appropriate
    list."""

    def clue(X: list[ArithRef]) -> BoolRef:
        return Or([equals(X, num) for num in lists[len(X)]])

    return clue


def compute_square_lists(n: int) -> list[list[int]]:
    """Generate lists of square numbers with n or fewer digits, excluding
    numbers containing zero."""
    lists = [[] for _ in range(N + 1)]
    odd = 1
    num = 1
    while num < 10**n:
        if "0" not in str(num):
            lists[len(str(num))].append(num)
        odd += 2
        num += odd
    return lists


def compute_fibonacci_lists(n: int) -> list[list[int]]:
    """Generate lists of Fibonacci numbers with n or fewer digits, excluding
    numbers containing zero."""
    lists = [[] for _ in range(N + 1)]
    a, b = 0, 1
    while b < 10**n:
        if "0" not in str(b):
            lists[len(str(b))].append(b)
        a, b = b, a + b
    return lists


def compute_prime_lists(n: int) -> list[list[int]]:
    """Generate lists of prime numbers with n or fewer digits, excluding
    numbers containing zero."""
    lists = [[] for _ in range(N + 1)]
    for k in range(n + 1):
        primes = primerange(10 ** (k - 1), 10**k)
        lists[k].extend([p for p in primes if "0" not in str(p)])
    return lists


def compute_divisible_by_digits_lists(n: int) -> list[list[int]]:
    """Generate lists of numbers with n or fewer digits that are divisible by
    all their digits."""
    lists = [[] for _ in range(N + 1)]
    for k in range(n + 1):
        nums = np.arange(10 ** (k - 1), 10**k)
        nums_str = nums.astype(str).flatten()
        digits = np.array([np.char.find(nums_str, str(d)) >= 0 for d in range(10)]).T
        nums_multiple_of_digits = nums.reshape(-1, 1) % np.arange(1, 10) == 0
        valid = ~digits[:, 0] & np.all(~digits[:, 1:] | nums_multiple_of_digits, axis=1)
        lists[k].extend(nums[valid])
    return lists


square_lists = compute_square_lists(6)
fibonacci_lists = compute_fibonacci_lists(N)
prime_lists = compute_prime_lists(3)
divisible_by_digits_lists = compute_divisible_by_digits_lists(7)


square = clue_from_lists(square_lists)
fibonacci = clue_from_lists(fibonacci_lists)
prime = clue_from_lists(prime_lists)
divisible_by_its_digits = clue_from_lists(divisible_by_digits_lists)


clues = [
    square,
    product_of_digits_is_20,
    multiple_of_13,
    multiple_of_32,
    divisible_by_its_digits,
    product_of_digits_is_25,
    divisible_by_its_digits,
    odd_palindrome,
    fibonacci,
    product_of_digits_is_2025,
    prime,
]


for i, clue_fn in tqdm(enumerate(clues), desc="Processing rows", total=len(clues)):
    for l in tqdm(range(N), desc=f"Processing row {i} with clue {clue_fn.__name__}", leave=False):
        for r in tqdm(range(l + 2, N + 1), desc=f"Processing numbers starting at {l}", leave=False):
            s += Implies(is_number(i, l, r), clue_fn(Y[i, l:r]))


# Solve
# ----------------------------------------------------------------------


def plot_solution(
    regions: NDArray[np.int32],
    highlighted: NDArray[np.bool_],
    clues: list[str],
    xm: NDArray[np.int32],
    ym: NDArray[np.int32],
) -> None:
    """Plot the puzzle solution showing the initial and final grids with region
    borders and highlighted cells."""
    # Create figure with extra space on the right for clues
    fig = plt.figure(figsize=(24, 11))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 1], wspace=1 / N)  # Add one-cell spacing
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Flip all arrays vertically to match the correct orientation
    xm = np.flipud(xm)
    ym = np.flipud(ym)
    highlighted = np.flipud(highlighted)
    regions = np.flipud(regions)

    # Create annotations
    xm_annot = np.array(xm, dtype=str)
    ym_annot = np.array(ym, dtype=str)
    xm_annot[xm == 0] = ""
    ym_annot[ym == 0] = ""

    # Create color arrays (-1 for black tiles, 1 for highlighted, 0 for white)
    xm_colors = np.zeros_like(xm)
    ym_colors = np.zeros_like(ym)
    xm_colors[highlighted] = 1
    ym_colors[highlighted] = 1
    xm_colors[xm == 0] = -1
    ym_colors[ym == 0] = -1

    # Custom colormap: black -> white -> yellow
    colors = ["black", "white", "yellow"]
    cmap = ListedColormap(colors)

    # Common arguments for both heatmaps
    heatmap_kwargs = {
        "fmt": "",
        "cmap": cmap,
        "cbar": False,
        "linewidths": 0.5,
        "linecolor": "black",
        "square": True,
        "center": 0,
        "vmin": -1,
        "vmax": 1,
        "xticklabels": False,
        "yticklabels": False,
    }

    # Plot heatmaps
    sns.heatmap(xm_colors, annot=xm_annot, ax=ax1, **heatmap_kwargs)
    g2 = sns.heatmap(ym_colors, annot=ym_annot, ax=ax2, **heatmap_kwargs)

    # Color the annotations in the second heatmap
    for i, j in product(range(N), repeat=2):
        if ym[i, j] != 0:  # Only process non-empty cells
            text = g2.texts[i * N + j]
            if ym[i, j] != xm[i, j]:
                text.set_color("red")
            else:
                text.set_color("black")

    # Add bold outline to each grid
    line_width = 2  # Increased line width
    for ax in [ax1, ax2]:
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)

        # Draw thick lines around the grid edges
        ax.axhline(y=0, color="black", linewidth=line_width, clip_on=False)
        ax.axhline(y=N, color="black", linewidth=line_width, clip_on=False)
        ax.axvline(x=0, color="black", linewidth=line_width, clip_on=False)
        ax.axvline(x=N, color="black", linewidth=line_width, clip_on=False)

        # Draw region borders
        for (i, j), region in np.ndenumerate(regions):
            if j < N - 1 and regions[i, j + 1] != region:
                ax.plot([j + 1, j + 1], [i, i + 1], "black", linewidth=2)
            if i < N - 1 and regions[i + 1, j] != region:
                ax.plot([j, j + 1], [i + 1, i + 1], "black", linewidth=2)

    # Add titles and adjust layout
    ax1.set_title("(after initial placement)")
    ax2.set_title("(after tiles placed; altered values in red)")

    # Add clues to the right of the second grid
    for i, clue in enumerate(reversed(clues)):
        ax2.text(N + 0.5, i + 0.5, f"{clue}", va="center")

    plt.tight_layout()
    plt.show()


def numbers_in_array(ym: NDArray[np.int32]) -> list[int]:
    """Extract all non-zero numbers from the grid, where numbers are formed by
    consecutive non-zero digits."""
    nums = []
    for ys in ym:
        row_nums = [int(x) for x in "".join(map(str, ys)).split("0") if x]
        nums.extend(row_nums)
    return nums


with Timer(initial_text="Checking z3 model"):
    num_solutions = 0
    while s.check() == sat:
        num_solutions += 1

        m = s.model()
        xm = evaluate_vars(m, X)
        tm = evaluate_vars(m, T)
        cm = evaluate_vars(m, C)
        dm = evaluate_vars(m, D)
        ym = evaluate_vars(m, Y)

        # Properties of the solution
        assert np.sum(xm) == np.sum(ym)
        assert np.all(ym == np.where(tm == 1, 0, xm + cm))
        assert np.sum(xm * tm) == np.sum(cm)

        nums = numbers_in_array(ym)
        duplicates = len(nums) != len(set(nums))

        if duplicates:
            print(f"Found solution {num_solutions:02d}, but it has duplicate numbers")
            s += Not(And([y == y_prev for y, y_prev in zip(Y.flat, ym.flat)]))
        else:
            print(f"Found solution {num_solutions:02d}, which has no duplicates!")
            print_array(xm, "X")
            print_array(ym, "Y")
            print(f"Answer: {sum(nums)}")
            break

clue_names = [
    "square",
    "product of digits is 20",
    "multiple of 13",
    "multiple of 32",
    "divisible by each of its digits",
    "product of digits is 25",
    "divisible by each of its digits",
    "odd palindrome",
    "fibonacci",
    "product of digits is 2025",
    "prime",
]
plot_solution(regions, highlighted, clue_names, xm, ym)
