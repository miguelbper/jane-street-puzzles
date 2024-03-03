# Imports
# ----------------------------------------------------------------------
from typing import Optional
from copy import deepcopy
from pprint import pprint
from time import time
from scipy.ndimage import label, sum_labels
from functools import reduce
from operator import mul
from itertools import product

'''
Blog post with detailed explanation of this solution:
https://miguelbper.github.io/2023/03/22/js-2023-02-twenty-four-seven-four-in-one.html
'''


# Types
# ----------------------------------------------------------------------

Board = list[list[int]]    # matrix containing board
Choices = list[list[int]]  # bitmask for possible values


# Data
# ----------------------------------------------------------------------

n = -1
matrix = [
    [n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, 1, n, n, n, n, n, 6, 5, n],
    [n, n, n, n, n, 3, n, n, n, n, 6, n],
    [n, 4, n, n, n, n, n, n, 7, n, n, n],
    [n, n, n, n, 2, n, n, n, n, n, n, 7],
    [n, n, 6, n, n, n, n, n, 3, 7, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, 7, n, 5, n, n, n, n, n, n],
    [n, 5, n, n, n, 7, n, n, n, n, n, n],
    [n, 6, 7, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, 6, n, n, n, n, n, n, n],
]

blue = [
    [5,  7,  7, 33, 29, 2, 40, 28, n, n, 36, n],  # row
    [6,  n,  n,  4,  n, n,  n,  n, n, n,  n, 5],  # col rev
    [4,  n,  n,  1,  n, n,  n,  n, n, n,  n, 7],  # row rev
    [6, 36, 30, 34, 27, 3, 40, 27, n, n,  7, n],  # col
]


# Functions
# ----------------------------------------------------------------------

# count[n] = number of 1s in binary representation of n.
count = [0 for _ in range(1 << 8)]
for n in range(1, 1 << 8):
    count[n] = 1 + count[n & (n - 1)]


# value[n] = location of first 1 in binary representation of n.
value = [-1 for _ in range(1 << 8)]
for n in range(1, 1 << 8):
    value[n] = next(v for v in range(8) if n & (1 << v))


# Conversion between board and choices
def board(cm: Choices) -> Board:
    ''' Given matrix of choices, return matrix of values. '''
    x = lambda c: value[c] if count[c] == 1 else -1
    return [[x(cm[i][j]) for j in range(12)] for i in range(12)]


def choices(xm: Board) -> Choices:
    ''' Given matrix of values, return matrix of choices. '''
    c = lambda x: 255 if x == -1 else 1 << x
    return [[c(xm[i][j]) for j in range(12)] for i in range(12)]


# Backtracking algorithm
def solution(xm: Board) -> Optional[Board]:
    ''' Main function of backtracking algorithm. Given initial board xm,
    compute filled board. Return None if no solution exists. '''
    stack = [choices(xm)]
    while stack:
        cm = prune(stack.pop())
        if reject(cm):
            continue
        if accept(cm):
            return board(cm)
        stack += expand(cm)
    return None


def accept(cm: Choices) -> bool:
    ''' True iff the board is completely filled. '''
    return all(count[cm[i][j]] == 1 for i, j in product(range(12), repeat=2))


def reject(cm: Choices) -> bool:
    ''' True iff a partially filled board will never lead to a sol. '''
    return blocked(cm) or not connected(cm)


def blocked(cm: Choices) -> bool:
    ''' True iff there is a cell s.t. no 0 <= x <= 7 is in the cell. '''
    return any(cm[i][j] == 0 for i, j in product(range(12), repeat=2))


def connected(cm: Choices) -> bool:
    ''' True iff there is a chance the partially filled board will lead
    to a connected board. '''
    mat = [[int(cm[i][j] != 1) for j in range(12)] for i in range(12)]
    _, num_components = label(mat)
    return num_components <= 1


def expand(cm: Choices) -> list[Choices]:
    ''' Given a partially filled board, choose the cell with lowest num
    of possible values. Return list of copies of the board, where in the
    cell, we replace by each possible value. '''
    # find (a, b) such that count[cm[a][b]] is minimal
    a, b = 0, 0
    m = count[cm[a][b]]
    for i, j in product(range(12), repeat=2):
        if m <= 1 or (1 < count[cm[i][j]] < m):
            a, b = i, j
            m = count[cm[a][b]]
        if m == 2:
            break

    # return list of copies of cm, but replacing cm[a][b] by each value
    ans = []
    for x in range(8):
        if cm[a][b] & (1 << x):
            cm_new = deepcopy(cm)
            cm_new[a][b] = 1 << x
            ans.append(cm_new)

    return ans


def prune(cm: Choices) -> Choices:
    ''' Given partially filled board, use constraints of the problem to
    remove numbers from the list of possibilities of each cell. '''
    ans = deepcopy(cm)

    # prune based on 2x2
    # ------------------------------------------------------------------
    for i, j in product(range(11), repeat=2):
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
                ans[a][b] &= 1

    # prune based on one 1, ..., seven 7
    # ------------------------------------------------------------------
    for grid in range(4):
        i0 = 5 if ((grid + 1) % 4 > 1) else 0
        i1 = 12 if ((grid + 1) % 4 > 1) else 7
        j0 = 5 if grid % 4 > 1 else 0
        j1 = 12 if grid % 4 > 1 else 7

        coordinates = {x: [] for x in range(8)}
        for i, j in product(range(i0, i1), range(j0, j1)):
            if count[ans[i][j]] == 1:
                coordinates[value[ans[i][j]]].append((i, j))

        for x in range(1, 8):
            # If there are x or more cells with only possibility x,
            # x is not in the other cells
            if len(coordinates[x]) >= x:
                for i, j in product(range(i0, i1), range(j0, j1)):
                    if (i, j) not in coordinates[x][:x]:
                        ans[i][j] &= (1 << x) ^ 255

    # prune based on row/col sum = 20, 4 nums
    # ------------------------------------------------------------------
    for grid in range(4):
        i0 = 5 if ((grid + 1) % 4 > 1) else 0
        i1 = 12 if ((grid + 1) % 4 > 1) else 7
        j0 = 5 if grid % 4 > 1 else 0
        j1 = 12 if grid % 4 > 1 else 7

        rows = [[(i, j) for j in range(j0, j1)] for i in range(i0, i1)]
        cols = [[(i, j) for i in range(i0, i1)] for j in range(j0, j1)]
        arrs = rows + cols

        for arr in arrs:
            cms = [ans[i][j] for (i, j) in arr]

            num_zeros = sum(value[c] == 0 for c in cms if count[c] == 1)
            num_posit = sum(value[c] >= 1 for c in cms if count[c] == 1)
            n = 4 - num_posit
            s = 20 - sum(value[c] for c in cms if count[c] == 1)

            if not (num_zeros <= 3 and num_posit <= 4) or (n == 0 and s != 0):
                return [[0 for _ in range(12)] for _ in range(12)]

            # loop over unknown cells
            for (i, j), c in zip(arr, cms):
                if count[c] > 1:
                    # loop over x s.t. x is not part of partition
                    for x in range(1, 8):
                        if n and not (n - 1 <= s - x <= 7*(n - 1)):
                            ans[i][j] &= (1 << x) ^ 255

    # prune based on bluenum sum
    # ------------------------------------------------------------------
    rows = [(blue[0][i], [(i, j) for j in range(5, 7)]) for i in range(12)]
    cols = [(blue[3][j], [(i, j) for i in range(5, 7)]) for j in range(12)]
    arrs = rows + cols

    for bluenum, arr in arrs:
        if bluenum <= 7:
            continue

        cms = [ans[i][j] for (i, j) in arr]
        n = sum(count[c] != 1 for c in cms)
        s = 40 - bluenum - sum(value[c] for c in cms if count[c] == 1)

        if n == 0 and s != 0:
            return [[0 for _ in range(12)] for _ in range(12)]

        # loop over unknown cells
        for (i, j), c in zip(arr, cms):
            if count[c] > 1:
                # loop over x s.t. x is not part of partition
                for x in range(1, 8):
                    if n and not (0 <= s - x <= 7*(n - 1)):
                        ans[i][j] &= (1 << x) ^ 255

    # prune based on bluenum line of sight
    # ------------------------------------------------------------------
    for side, a in product(range(4), range(12)):
        bluenum = blue[side][a]
        if 1 <= bluenum <= 7:
            for b in range(12):
                i = a if not side % 2 else (b if side == 3 else 11 - b)
                j = a if side % 2 else (b if not side else 11 - b)
                if ans[i][j] != 1:
                    # remove all except {0, bluenum} from ans[i][j]
                    for x in range(1, 8):
                        if x != bluenum:
                            ans[i][j] &= (1 << x) ^ 255
                    break

    return ans


# Solution
# ----------------------------------------------------------------------

t0 = time()
sol = solution(matrix)
t1 = time() - t0
print(f'\nSolved in {t1:.2f} sec. Solution = ')
pprint(sol)
'''
Solved in 1.50 sec. Solution =
    [0 5 0 6 0 3 6 0 0 0 7 4]
    [0 7 7 1 0 0 5 0 4 6 5 0]
    [0 0 0 7 5 3 5 0 6 0 6 0]
    [6 4 3 0 0 7 0 5 7 1 0 0]
    [7 0 0 0 2 7 4 2 0 0 0 7]
    [2 0 6 6 6 0 0 6 3 7 0 4]
    [5 4 4 0 7 0 0 7 0 6 2 5]
    [7 0 0 0 1 5 7 1 0 0 7 0]
    [0 5 3 7 0 5 0 0 5 0 4 6]
    [6 5 0 0 0 7 2 0 6 0 0 5]
    [0 6 7 3 0 0 4 6 6 4 0 0]
    [0 0 0 4 6 3 7 0 0 3 7 0]
'''


# Product of areas
# ----------------------------------------------------------------------
def areas(xm: Board) -> int:
    ''' Product of the areas of the unfilled regions. '''
    mat = [[int(xm[i][j] == 0) for j in range(12)] for i in range(12)]
    labels, n = label(mat)
    area = sum_labels(mat, labels, index=range(1, n + 1))
    return int(reduce(mul, area))


print(f'\nans = {areas(sol)}')  # type: ignore
# ans = 74649600
