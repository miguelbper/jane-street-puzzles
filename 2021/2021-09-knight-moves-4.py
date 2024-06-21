'''
Let
    m = side of the square grid = 10
    k = number of regions = 17

    n = number of steps the knight takes
    s = sum of every number in the grid
    r = sum of the numbers in a region of the grid

Then,
    s = n*(n+1) / 2
    s = 17 * r
    n <= 100
    r >= (12 + 5 + 8 + 2) + (11 + 13) = 51

Lem. r <= 2*n - 3.
Pf. Consider a region with two squares. The biggest sum that region can
have is n + (n - 3) = 2*n - 3.

Lem. If r is even, then r <= n.
Pf. Consider a region with two squares. If the knight were to visit that
region twice, then r would be odd (sum of an even and odd number). So,
the knight visits that region exactly once.

These facts taken together imply that
    n = 50
    r = 75
    s = 1275
'''

# Imports
# ----------------------------------------------------------------------
from itertools import product
import numpy as np
from copy import deepcopy


# Data
# ----------------------------------------------------------------------

regions = [
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  1,  0,  0,  0,  0,  2,  0],
    [ 0,  0,  0,  1,  0,  0,  0,  0,  2,  0],
    [ 0,  0,  1,  1,  1,  1,  3,  2,  2,  2],
    [ 5,  0,  1,  0,  0,  4,  3,  3,  3,  3],
    [ 5,  0,  0,  0,  4,  4,  4, 10, 10,  3],
    [ 5,  6,  6,  7,  4,  9,  9, 10, 10, 11],
    [ 5, 12,  6,  7,  7,  8,  8,  8, 10, 11],
    [ 5, 12, 13,  7, 16, 16, 16, 15, 15, 11],
    [12, 12, 13, 14, 14, 14, 14, 14, 15, 11],
]

grid = [
    [12, 0, 0,  0,  0, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0,  0, 0, 5, 0, 23,  0],
    [ 0, 0, 0,  0,  0, 0, 8, 0,  0,  0],
    [ 0, 0, 0, 14,  0, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
    [ 0, 2, 0,  0,  0, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0, 20, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0, 33, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
    [ 0, 0, 0,  0,  0, 0, 0, 0,  0, 28],
]

directions = [
    ( 2,  1),
    ( 1,  2),
    (-1,  2),
    (-2,  1),
    (-1, -2),
    ( 1, -2),
    ( 2, -1),
    (-2, -1),
]

m = len(grid)
k = 1 + np.max(regions)
n = 50
s = n * (n + 1) // 2
r = s // k

coords = {r: [] for r in range(k)}
for i, j in product(range(m), repeat=2):
    coords[regions[i][j]].append((i, j))

stack = []
i, j = 5, 1
state = (grid, (i, j), 2)
for x, y in directions:
    x += i
    y += j
    if not (0 <= x < m and 0 <= y < m):
        continue
    new_grid = deepcopy(grid)
    new_grid[x][y] = 1
    stack.append((new_grid, (x, y), 1))


# Solution
# ----------------------------------------------------------------------

while stack:
    state = stack.pop()
    grid, coor, step = state
    a, b = coor
    reg_sum = [sum(grid[i][j] for i, j in coords[l]) for l in range(k)]

    # If state is solution, break
    # ------------------------------------------------------------------
    if step == n and all(r == reg_sum[l] for l in range(k)):
        break

    # If state is blocked, continue
    # ------------------------------------------------------------------

    # if grid has a duplicate, block
    seen = set()
    flag = False
    for i, j in product(range(m), repeat=2):
        g = grid[i][j]
        if g in seen:
            flag = True
            break
        elif g:
            seen.add(g)
    if flag:
        continue

    # if there is a region l s.t. its region sum will never be r, block
    flag = False
    for l in range(k):
        sl = reg_sum[l]
        # sum > r => sum will never be r
        if sl > r:
            flag = True
            break
        # r > sum >= r - step => sum will never be r (next step is 'step + 1')
        if r > sl >= r - step:
            flag = True
            break
        # sum != r and (final step or region is filled) => sum will never be r
        if sl != r and (step == n or all(grid[i][j] for i, j in coords[l])):
            flag = True
            break
    if flag:
        continue

    # Compute new states and add them to the stack
    # ------------------------------------------------------------------
    for x, y in directions:
        x += a
        y += b
        if not (0 <= x < m and 0 <= y < m):
            continue
        if grid[x][y] not in {0, step + 1}:
            continue
        new_grid = deepcopy(grid)
        new_grid[x][y] = step + 1
        stack.append((new_grid, (x, y), step + 1))


# Output
# ----------------------------------------------------------------------

grid, coor, step = state
ans = sum(max(row)**2 for row in grid)
print(f'answer = {ans}\ngrid   = ')
for row in grid:
    print('[' + ' '.join(f'{x:2d}' for x in row) + ']')

'''
answer = 14820
grid   =
[12  0  0  0  0  9  0  7  0  0]
[ 0  0 13 10  0  0  5  0 23  0]
[ 0 11  0 17  4  0  8  0  6  0]
[ 1  0  0 14  0 18  0 22  0 24]
[ 0  0 16  3  0 21 50 25  0  0]
[ 0  2  0  0 15  0 19 48  0  0]
[ 0 41 34  0 20 49 26  0  0 47]
[35 38  0 42 33 30 45  0 27  0]
[40  0 36  0  0 43 32 29 46  0]
[37  0 39  0 31  0  0 44  0 28]
'''
