from itertools import product
from functools import reduce
from operator import mul
from typing import Optional
from z3 import Solver, Int, And, Or, Not, Sum, PbEq, sat, BoolRef
from scipy.ndimage import label, sum_labels
from pprint import pprint
from codetiming import Timer


# Parameters
# ----------------------------------------------------------------------
Board = list[list[int]]
Coord = tuple[int, int]

z = -1
matrix = [
    [z, z, z, z, z, z, z, z, z, z, z, z],
    [z, z, z, 1, z, z, z, z, z, 6, 5, z],
    [z, z, z, z, z, 3, z, z, z, z, 6, z],
    [z, 4, z, z, z, z, z, z, 7, z, z, z],
    [z, z, z, z, 2, z, z, z, z, z, z, 7],
    [z, z, 6, z, z, z, z, z, 3, 7, z, z],
    [z, z, z, z, z, z, z, z, z, z, z, z],
    [z, z, z, z, z, z, z, z, z, z, z, z],
    [z, z, z, 7, z, 5, z, z, z, z, z, z],
    [z, 5, z, z, z, 7, z, z, z, z, z, z],
    [z, 6, 7, z, z, z, z, z, z, z, z, z],
    [z, z, z, z, 6, z, z, z, z, z, z, z],
]

blue = [
    [5,  7,  7, 33, 29, 2, 40, 28, z, z, 36, z],  # row
    [6,  z,  z,  4,  z, z,  z,  z, z, z,  z, 5],  # col rev
    [4,  z,  z,  1,  z, z,  z,  z, z, z,  z, 7],  # row rev
    [6, 36, 30, 34, 27, 3, 40, 27, z, z,  7, z],  # col
]

R = list(range(12))
IJ = list(product(R, repeat=2))


# Utility functions
# ----------------------------------------------------------------------
def neighbours(i: int, j: int) -> list[Coord]:
    ''' Return the coordinates of the neighbours of (i, j). '''
    direc = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neigh = [(i + di, j + dj) for di, dj in direc]
    return [(x, y) for x, y in neigh if 0 <= x < 12 and 0 <= y < 12]


def connected(xm: Board) -> bool:
    ''' True iff the board is connected. '''
    return label(xm)[1] <= 1


def areas(xm: Board) -> int:
    ''' Product of the areas of the unfilled regions. '''
    mat = [[int(xm[i][j] == 0) for j in R] for i in R]
    labels, k = label(mat)
    area = sum_labels(mat, labels, index=range(1, k + 1))
    return int(reduce(mul, area))


def grid_ranges(grid_num: int) -> tuple[int, int, int, int]:
    ''' Return index bounds for a given 7x7 grid index. '''
    i0 =  5 if ((grid_num + 1) % 4 > 1) else 0
    i1 = 12 if ((grid_num + 1) % 4 > 1) else 7
    j0 =  5 if grid_num % 4 > 1 else 0
    j1 = 12 if grid_num % 4 > 1 else 7
    return i0, i1, j0, j1


def rowcol_coords(side_idx: int, rowcol_idx: int) -> list[Coord]:
    ''' Return the coordinates of the row or column. '''
    is_row = not (side_idx % 2)      # True if rowcol_idx is the idx of a row
    is_inc = (side_idx + 1) % 4 < 2  # True if the other idx is increasing
    idx_range = R if is_inc else R[::-1]
    coord = lambda idx: (rowcol_idx, idx) if is_row else (idx, rowcol_idx)
    coords = [coord(idx) for idx in idx_range]
    return coords


# Solver & constraints
# ----------------------------------------------------------------------
X = [[Int(f'x_[{i}, {j}]') for j in R] for i in R]
s = Solver()

# Ranges for each variable
s.add([And(0 <= X[i][j], X[i][j] <= 7) for i, j in IJ])

# Given numbers
for i, j in IJ:
    if matrix[i][j] != z:
        s.add(X[i][j] == matrix[i][j])

# Each 7-by-7 grid should contain one 1, two 2’s, etc., up to seven 7’s
for grid_num in range(4):
    i0, i1, j0, j1 = grid_ranges(grid_num)
    grid = list(product(range(i0, i1), range(j0, j1)))
    for x in range(1, 8):
        s.add(PbEq([(X[i][j] == x, 1) for i, j in grid], x))

# Each row/col within the 7-by-7’s must contain exactly 4 nums which sum to 20
for grid_num in range(4):
    i0, i1, j0, j1 = grid_ranges(grid_num)
    # rows
    for i in range(i0, i1):
        s.add(PbEq([(X[i][j] != 0, 1) for j in range(j0, j1)], 4))
        s.add(Sum([X[i][j] for j in range(j0, j1)]) == 20)
    # cols
    for j in range(j0, j1):
        s.add(PbEq([(X[i][j] != 0, 1) for i in range(i0, i1)], 4))
        s.add(Sum([X[i][j] for i in range(i0, i1)]) == 20)

# Every 2-by-2 subsquare in the grid must contain at least one empty cell
for i, j in product(range(11), repeat=2):
    dirs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    s.add(Or([X[i + di][j + dj] == 0 for di, dj in dirs]))

# A number <= 7 represents the value of the first number it sees
def first_num_seen(coords: list[Coord], blue_num: int, k: int) -> BoolRef:
    zeros_before_k = And([X[i][j] == 0 for i, j in coords[:k]])
    first_num = X[coords[k][0]][coords[k][1]] == blue_num
    return And(zeros_before_k, first_num)

for side_idx, side_nums in enumerate(blue):
    for rowcol_idx, blue_num in enumerate(side_nums):
        if z < blue_num <= 7:
            coords = rowcol_coords(side_idx, rowcol_idx)
            s.add(Or([first_num_seen(coords, blue_num, k) for k in range(4)]))

# A number > 7 represents the sum of the row or column it is facing
for side_idx, side_nums in enumerate(blue):
    for rowcol_idx, blue_num in enumerate(side_nums):
        if blue_num > 7:
            coords = rowcol_coords(side_idx, rowcol_idx)
            s.add(Sum([X[i][j] for i, j in coords]) == blue_num)

# The numbered cells must form a connected region
# Wrap solver in a function which also checks connectivity
def solve(s: Solver) -> Optional[Board]:
    while s.check() == sat:
        m = s.model()
        xm = [[m.evaluate(X[i][j]).as_long() for j in R] for i in R]
        if connected(xm):
            return [[xm[i][j] for j in R] for i in R]
        else:
            s.add(Not(And([X[i][j] == xm[i][j] for i, j in IJ])))
    return None


# Solve problem
# ----------------------------------------------------------------------
with Timer():
    xm = solve(s)

if xm:
    ans = areas(xm)
    print(f'ans = {areas(xm)}\nboard = ')
    pprint(xm)
else:
    print('No solution found.')
'''
Elapsed time: 0.1507 seconds
ans = 74649600
board =
[[0, 5, 0, 6, 0, 3, 6, 0, 0, 0, 7, 4],
 [0, 7, 7, 1, 0, 0, 5, 0, 4, 6, 5, 0],
 [0, 0, 0, 7, 5, 3, 5, 0, 6, 0, 6, 0],
 [6, 4, 3, 0, 0, 7, 0, 5, 7, 1, 0, 0],
 [7, 0, 0, 0, 2, 7, 4, 2, 0, 0, 0, 7],
 [2, 0, 6, 6, 6, 0, 0, 6, 3, 7, 0, 4],
 [5, 4, 4, 0, 7, 0, 0, 7, 0, 6, 2, 5],
 [7, 0, 0, 0, 1, 5, 7, 1, 0, 0, 7, 0],
 [0, 5, 3, 7, 0, 5, 0, 0, 5, 0, 4, 6],
 [6, 5, 0, 0, 0, 7, 2, 0, 6, 0, 0, 5],
 [0, 6, 7, 3, 0, 0, 4, 6, 6, 4, 0, 0],
 [0, 0, 0, 4, 6, 3, 7, 0, 0, 3, 7, 0]]
'''
