# Imports
# ----------------------------------------------------------------------
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from itertools import product
from math import gcd
from operator import mul
from pprint import pprint
from time import time

from scipy.ndimage import label, sum_labels
from tqdm import tqdm

# Types
# ----------------------------------------------------------------------
# fmt: off
Board = list[list[int]]                        # matrix containing board
Choices = list[list[int]]                      # bitmask for possible x
Orientation = int                              # orientation of the hook
Number = int                                   # number in the hook
HookConfig = list[tuple[Orientation, Number]]  # hook configuration
Square = tuple[int, int, int, int]
# fmt: on

# Parameters & Utility functions
# ----------------------------------------------------------------------
row_gcd = [55, 1, 6, 1, 24, 3, 6, 7, 2]
col_gcd = [5, 1, 6, 1, 8, 1, 22, 7, 8]
n = len(row_gcd)
hook_len = [2 * (n - i) - 1 for i in range(n)]

m = 2  # m = number of different values that could be in a cell
# bits & retain[x] = (new bitmask where everything except x is set to 0)
retain = [1 << x for x in range(m)]

# count[n] = number of 1s in binary representation of n.
count = [0 for _ in range(1 << m)]
for i in range(1, 1 << m):
    count[i] = 1 + count[i & (i - 1)]

# value[n] = location of first 1 in binary representation of n.
value = [-1 for _ in range(1 << m)]
for i in range(1, 1 << m):
    value[i] = next(v for v in range(m) if i & (1 << v))


# Conversion between board and choices
def board(cm: Choices) -> Board:
    """Given matrix of choices, return matrix of values."""
    x = lambda c: value[c] if count[c] == 1 else -1
    return [[x(cm[i][j]) for j in range(n)] for i in range(n)]


def choices(xm: Board) -> Choices:
    """Given matrix of values, return matrix of choices."""
    c = lambda x: (1 << m) - 1 if x == -1 else 1 << x
    return [[c(xm[i][j]) for j in range(n)] for i in range(n)]


# Utility functions
def transpose(xm: Board) -> Board:
    """Returns transpose of the matrix xm."""
    return list(map(list, zip(*xm)))


def mygcd(xs: list[int]) -> int:
    """Returns the greatest common divisor of the list xs."""
    if not xs:
        return 0
    return reduce(gcd, xs)


# Find valid sets of orientations and numbers
# ----------------------------------------------------------------------


def find_hooks(init_hook: HookConfig) -> list[HookConfig]:
    """Return list of "admissible" hook configurations.

    A hook configuration is admissible if     (1) x[i] <= hook_len[i]
    for all i     (2) For every row/col, it is possible to fill that
    row/col with         the numbers prescribed by the hook
    configuration s.t. the         gcd constraint of the puzzle is
    satisfied. Computed using a backtracking algorithm.
    """
    stack: list[HookConfig] = [init_hook]
    hooks: list[HookConfig] = []

    while stack:
        hook = stack.pop()
        if not valid_hook(hook):
            continue
        if len(hook) == n:
            hooks.append(hook)
        else:
            stack += expand_hook(hook)

    return hooks


def hook_matrix(hook: HookConfig) -> tuple[Board, Square]:
    """Example:

    >>> hook = [(0, 5), (2, 8), (2, 7), (0, 9)]
    >>> matrix, (i0, j0, i1, j1) = hook_matrix(hook)
    >>> pprint(matrix)
    >>> print(f'{(i0, j0, i1, j1) = }')

    [[5, 5, 5, 5, 5, 5, 5, 5, 5],
     [5, 9, 9, 9, 9, 9, 9, 7, 8],
     [5, 9, 0, 0, 0, 0, 0, 7, 8],
     [5, 9, 0, 0, 0, 0, 0, 7, 8],
     [5, 9, 0, 0, 0, 0, 0, 7, 8],
     [5, 9, 0, 0, 0, 0, 0, 7, 8],
     [5, 9, 0, 0, 0, 0, 0, 7, 8],
     [5, 7, 7, 7, 7, 7, 7, 7, 8],
     [5, 8, 8, 8, 8, 8, 8, 8, 8]]
    (i0, j0, i1, j1) = (2, 2, 7, 7)
    """
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    i0, j0, i1, j1 = 0, 0, n, n

    for o, x in hook:
        o_is_01 = o < 2
        o_is_03 = ((o + 1) % 4) < 2

        # in matrix, fill hook determined by o with x
        row_ind = i0 if o_is_03 else i1 - 1
        col_ind = j0 if o_is_01 else j1 - 1
        row = [(row_ind, j) for j in range(j0, j1)]
        col = [(i, col_ind) for i in range(i0, i1)]
        arr = row + col
        for i, j in arr:
            matrix[i][j] = x

        # update i0, j0, i1, j1
        i0 += o_is_03
        j0 += o_is_01
        i1 -= not o_is_03
        j1 -= not o_is_01

    return (matrix, (i0, j0, i1, j1))


def nums_subset(arr: list[int], s: int) -> list[int]:
    """Example:
    >>> arr = [5,9,6,4,3,1,2,7,8] #             596431279    | output:
    >>> s = 182                   # (182)_10 = (010110110)_2 | 9,43,27
    >>> nums = nums_subset(arr, s)
    >>> print(nums)
    [27, 43, 9]
    """
    nums = []
    curr = 0
    expn = 0
    for i in range(n):
        if (1 << i) & s:
            curr += arr[n - 1 - i] * 10**expn
            expn += 1
        else:
            if curr:
                nums.append(curr)
            curr = 0
            expn = 0
    if curr:
        nums.append(curr)
    return nums


def valid_hook(hook: HookConfig) -> bool:
    """True if for every row/col there exists a subset whose gcd is the given
    one."""

    rows, (i0, j0, i1, j1) = hook_matrix(hook)
    cols = transpose(rows)

    def check_arr(x: int, arr: list[int]) -> bool:
        return any(x == mygcd(nums_subset(arr, s)) for s in range(1, 1 << n))

    filled_rows = (i for i in range(n) if not (i0 <= i < i1))
    filled_cols = (j for j in range(n) if not (j0 <= j < j1))
    ans = all(check_arr(row_gcd[i], rows[i]) for i in filled_rows) and all(
        check_arr(col_gcd[j], cols[j]) for j in filled_cols
    )

    return ans


def expand_hook(hook: HookConfig) -> list[HookConfig]:
    """Add one hook to a given hook configuration.

    Output the list of all possible ways the extra hook can be added.
    """
    l = len(hook)
    used = [x for _, x in hook]

    # seen[x] == 1 <=> x in used
    seen = [0 for _ in range(n + 1)]
    for x in used:
        seen[x] = 1

    def valid(x: int) -> bool:
        if (x == 1 and l != n - 1) or (x == 2 and l != n - 2):
            return False
        unused = [y for y in range(n, 0, -1) if not seen[y] and y != x]
        nums = used + [x] + unused
        return all(nums[i] <= hook_len[i] for i in range(n))

    smallest_valid = next(x for x in range(1, n + 1) if not seen[x] and valid(x))

    xs = [x for x in range(smallest_valid, n + 1) if not seen[x]]
    os = range(4 if l < n - 1 else 1)
    ans = [hook + [(o, x)] for o in os for x in xs]

    return ans


# For each hook = list of orientations and numbers, try to find solution
# ----------------------------------------------------------------------


def solution(hook: HookConfig) -> Board | None:
    """Given a hook configuration, output a solution to the puzzle, if a
    solution exists, otherwise output None (Using backtracking)."""
    cm0 = [[(1 << m) - 1 for _ in range(n)] for _ in range(n)]
    stack = [cm0]
    while stack:
        cm = prune(hook, stack.pop())
        if reject(hook, cm):
            continue
        if accept(cm):
            return board(cm)
        stack += expand(cm)
    return None


def accept(cm: Choices) -> bool:
    """True if and only if the board is completely filled."""
    return all(count[cm[i][j]] == 1 for i, j in product(range(n), repeat=2))


def expand(cm: Choices) -> list[Choices]:
    """Given a partially filled board, choose a cell (a, b) whose value is not
    yet filled.

    Return list of copies of the board, where in the cell, we replace by
    each possible value.
    """

    # If m == 2, then count[cm[i][j]] > 1 <=> cm[i][j] == 3
    a, b = next(((i, j) for i, j in product(range(n), repeat=2) if cm[i][j] == 3))

    ans = []
    for x in range(m):
        cm_new = deepcopy(cm)
        cm_new[a][b] = 1 << x
        ans.append(cm_new)

    return ans


def prune(hook: HookConfig, cm: Choices) -> Choices:
    """Given partially filled board, use constraints of the problem to remove
    numbers from the list of possibilities of each cell."""
    inv = [[0 for _ in range(n)] for _ in range(n)]  # invalid board
    ans = deepcopy(cm)

    # prune based on 2x2 rule
    # ------------------------------------------------------------------
    for i, j in product(range(n - 1), repeat=2):
        for corner in range(4):
            a = i + ((corner + 1) % 4 > 1)
            b = j + (corner > 1)

            # if all cells except (a, b) do not have a 0,
            # then cell (a, b) can't be > 0.
            remove = True
            for corner_ in range(4):
                if corner_ != corner:
                    a_ = i + ((corner_ + 1) % 4 > 1)
                    b_ = j + (corner_ > 1)
                    remove &= not (ans[a_][b_] & 1)

            if remove:
                ans[a][b] &= retain[0]

    # prune based on x times x
    # ------------------------------------------------------------------
    i0, j0, i1, j1 = 0, 0, n, n

    for o, x in hook:
        o_is_01 = o < 2
        o_is_03 = ((o + 1) % 4) < 2

        # in matrix, fill hook determined by o with x
        row_ind = i0 if o_is_03 else i1 - 1
        col_ind = j0 if o_is_01 else j1 - 1
        row = [(row_ind, j) for j in range(j0, j1)]
        col = [(i, col_ind) for i in range(i0, i1)]
        arr = row + col

        # update i0, j0, i1, j1
        i0 += o_is_03
        j0 += o_is_01
        i1 -= not o_is_03
        j1 -= not o_is_01

        coordinates = defaultdict(set)
        for i, j in arr:
            coordinates[ans[i][j]].add((i, j))

        num_ones = len(coordinates[2])  # 2 = (1 << 1) = 2**1
        num_unkn = len(coordinates[3])  # If m == 2, then count[c]>1 <=> c==3
        num_miss = x - num_ones

        if not (0 <= num_miss <= num_unkn):
            return inv

        if num_miss in {0, num_unkn}:
            fill_with = int(bool(num_miss))
            for i, j in coordinates[3]:
                ans[i][j] &= retain[fill_with]

    return ans


def reject(hook: HookConfig, cm: Choices) -> bool:
    """True iff a partially filled board will never lead to a sol."""
    return blocked(cm) or not connected(cm) or not valid_gcd(hook, cm)


def blocked(cm: Choices) -> bool:
    """True iff there is a cell s.t.

    no 0 <= x < m is in the cell.
    """
    return any(cm[i][j] == 0 for i, j in product(range(n), repeat=2))


def connected(cm: Choices) -> bool:
    """True iff there is a chance the partially filled board will lead to a
    connected board."""
    mat = [[int(cm[i][j] != 1) for j in range(n)] for i in range(n)]
    _, num_components = label(mat)
    return num_components <= 1


def valid_gcd(hook: HookConfig, cm: Choices) -> bool:
    """True if every row/col "might" satisfy the gcd property.

    Meaning of "might satisfy the gcd property":
    1. Consider a partially filled row/col:
            arr = [5,?,0,4,3,0,2,?,8]
    2. Extract the numbers which are with certainty in that row/col. This
       means that we split on 0s:
            [5, ?], [4, 3], [2, ?, 8]
       and then remove lists with a '?':
            nums = [43]
    3. Then check that gcd(nums) is a multiple of given_gcd. If there
       were no '?' originally, then gcd(nums) == given_gcd.
    """
    xm = board(cm)
    mat, _ = hook_matrix(hook)
    rows_aux = lambda i, j: xm[i][j] * (1 if xm[i][j] == -1 else mat[i][j])
    rows = [[rows_aux(i, j) for j in range(n)] for i in range(n)]
    cols = transpose(rows)

    def check_arr(x: int, arr: list[int]) -> bool:
        gcd_ = mygcd(nums(arr))
        q, r = divmod(gcd_, x)
        full = all(a != -1 for a in arr)
        return r == 0 and (not full or q == 1)

    ans = all(check_arr(row_gcd[i], rows[i]) for i in range(n)) and all(
        check_arr(col_gcd[j], cols[j]) for j in range(n)
    )

    return ans


def nums(arr: list[int]) -> list[int]:
    """Similar to nums_subset, but with the possibility that values in arr are
    'unknown'."""
    ans = []
    read = True
    curr = 0
    expn = 0
    for i in reversed(range(n)):
        if arr[i] > 0:
            curr += read * (arr[i] * 10**expn)
            expn += read
        else:
            if arr[i] == 0 and curr:
                ans.append(curr)
            read = arr[i] == 0
            curr = 0
            expn = 0
    if curr:
        ans.append(curr)
    return ans


# When the board is computed, this gives the solution to the puzzle.
def areas(xm: Board) -> int:
    """Product of the areas of the unfilled regions."""
    mat = [[int(xm[i][j] == 0) for j in range(n)] for i in range(n)]
    labels, k = label(mat)
    area = sum_labels(mat, labels, index=range(1, k + 1))
    return int(reduce(mul, area))


# Solve
# ----------------------------------------------------------------------

# 1 (a) Compute all hook configurations using find_hooks (~90 sec)
# The configuration which leads to a solution has index 373 in the list
t0 = time()
hooks = find_hooks([])
t1 = time() - t0
num_hooks = len(hooks)
print(f"Found {num_hooks} hooks in {t1:.2f} sec.")
"""
Note: trying every hook configuration takes a long time. To try only
the hook configuration which leads to a solution, uncomment these lines
"""
# 1 (b) Define only the hook configuration which leads to a solution
hooks = [[(0, 5), (2, 8), (2, 7), (0, 9), (1, 6), (0, 4), (0, 3), (2, 2), (0, 1)]]
num_hooks = len(hooks)

# 2 For each hook configuration, try to solve the puzzle
brd = [[0 for _ in range(n)] for _ in range(n)]
sol = 0

for i in tqdm(range(num_hooks)):
    hk = hooks[i]
    xm = solution(hk)
    if xm:
        mat, _ = hook_matrix(hk)
        brd = [[xm[i][j] * mat[i][j] for j in range(n)] for i in range(n)]
        sol = areas(brd)
        break

print(f"\nSolution = {sol}\n")
pprint(brd)
"""Solution = 15552.

[[5, 5, 0, 0, 0, 5, 5, 0, 0],  [0, 9, 9, 9, 9, 0, 9, 0, 8],  [0, 0, 6,
0, 4, 4, 4, 7, 8],  [0, 9, 0, 4, 3, 0, 0, 0, 8],  [0, 9, 6, 0, 3, 1, 2,
0, 0],  [0, 9, 0, 0, 3, 0, 2, 7, 0],  [0, 9, 6, 6, 6, 6, 0, 7, 8],  [0,
7, 0, 7, 0, 7, 0, 7, 0],  [5, 8, 0, 8, 8, 8, 0, 0, 0]]
"""
