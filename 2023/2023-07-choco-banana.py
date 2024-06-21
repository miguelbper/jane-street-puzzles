# Imports
# ----------------------------------------------------------------------
from typing import Optional
from copy import deepcopy
from pprint import pprint
from itertools import product
from math import prod
from time import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Types
Grid = list[list[int]]     # matrix containing Grid
Choices = list[list[int]]  # bitmask for possible x

# Input
# ----------------------------------------------------------------------

z = 0

grid = [
    [ 6,  6,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  6,  6],
    [ 6,  z,  z,  z,  z,  z,  z,  z,  z,  8, 12,  z,  z,  z,  z,  z,  z,  z,  z,  6],
    [ z,  z,  z, 10, 10,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, 12, 12,  z,  z,  z],
    [ z,  z,  z, 10,  z,  z, 10, 10,  z,  z,  z,  z, 11, 11,  z,  z,  4,  z,  z,  z],
    [ z,  z,  z,  z,  z,  z, 10,  z,  z,  z,  z,  z,  z, 11,  z,  z,  z,  z,  z,  z],
    [ z, 15,  z,  z,  z,  z,  z,  z,  z,  3,  4,  z,  z,  z,  z,  z,  z,  z,  3,  z],
    [ z,  4,  z,  z,  z,  z,  z,  z,  z,  6,  5,  z,  z,  z,  z,  z,  z,  z, 12,  z],
    [ z,  z,  z,  z,  z,  z,  9,  z,  z,  z,  z,  z,  z,  8,  z,  z,  z,  z,  z,  z],
    [ z,  z,  z, 15,  z,  z,  9,  9,  z,  z,  z,  z,  8,  8,  z,  z,  8,  z,  z,  z],
    [ z,  z,  z,  1,  9,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  1,  7,  z,  z,  z],
    [ 4,  z,  z,  z,  z,  z,  z,  z,  z, 12,  8,  z,  z,  z,  z,  z,  z,  z,  z,  4],
    [ 4,  4,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  4,  4],
]

# grid = [
#     [z, 6, z, z, z, z, z],
#     [z, z, z, z, z, z, 3],
#     [z, z, z, 3, z, z, z],
#     [z, z, 6, 1, 5, z, z],
#     [z, z, z, 4, z, z, z],
#     [5, z, z, z, z, z, z],
#     [z, z, z, z, z, 4, z],
# ]


# Bitmasks
# ----------------------------------------------------------------------

''' Problem: fill each cell with a number x <- {0,...,b-1} such that
every constraint is satisfied.

At each stage of the algorithm, for each cell there is a set of numbers
that could be in that cell. When the algorithm starts, that set will be
{0,...,b-1}. Each subset S < {0,...,b-1} can be represented by a bitmask
c = Σ_{x <- S} 2^x.
'''

m, n = len(grid), len(grid[0])  # shape of the grid
b = 2                           # b = num_bits, x <- {0,...,b-1}
bit = [(1 << x) for x in range(b)]
B = 0  # black
W = 1  # white

# c &= remove[x] -> remove x from the bitmask c
remove = [(1 << x) ^ ((1 << b) - 1) for x in range(b)]

# c &= remove_except[x] -> remove everything except x from the bitmask c
remove_except = [1 << x for x in range(b)]

# count[c] = number of 1s in binary representation of c.
count = [0 for _ in range(1 << b)]
for c in range(1, 1 << b):
    count[c] = 1 + count[c & (c - 1)]

# value[c] = location of first 1 in binary representation of c.
value = [-1 for _ in range(1 << b)]
for c in range(1, 1 << b):
    value[c] = next(v for v in range(b) if c & (1 << v))


# Conversion between Grid and Choices
def board(cm: Choices) -> Grid:
    ''' Given matrix of choices, return matrix of values. '''
    x = lambda c: value[c] if count[c] == 1 else -1
    return [[x(cm[i][j]) for j in range(n)] for i in range(m)]


def choices(xm: Grid) -> Choices:
    ''' Given matrix of values, return matrix of choices. '''
    c = lambda x: (1 << b) - 1 if x == -1 else 1 << x
    return [[c(xm[i][j]) for j in range(n)] for i in range(m)]


M = max(map(max, grid))
divisors = [[d for d in range(1, x + 1) if x % d == 0] for x in range(M + 1)]
nums = []
for i, j in product(range(m), range(n)):
    if grid[i][j]:
        nums.append((i, j, grid[i][j]))
nums.sort(key=lambda t: t[2] * len(divisors[t[2]]))


# Algorithm
# ----------------------------------------------------------------------

def solution(xm: Grid) -> Optional[Grid]:
    ''' Main function of backtracking algorithm. Given initial grid xm,
    compute filled grid. Return None if no solution exists. '''
    stack = [choices(xm)]
    while stack:
        cm = prune(stack.pop())
        if not cm:
            continue
        if accept(cm):
            return board(cm)
        stack += expand(cm)
    return None


def accept(cm: Choices) -> bool:
    ''' True iff the grid is completely filled. '''
    return all(count[cm[i][j]] == 1 for i, j in product(range(m), range(n)))


def expand(cm: Choices) -> list[Choices]:
    ''' Start with a partially filled grid cm. Then
        1. Find a number in a cell which is unfilled or which is black
           and incomplete. Fill grid with all possibilities of black
           rectangles, return those possibilities. If such a number
           can't be found, then:
        2. Find an empty cell. Fill that cell with white or black.
    '''

    components, num_components = ccs(cm)
    _, complete = nbhds(cm, components, num_components)

    for i, j, x in nums:
        unknown = count[cm[i][j]] > 1
        black_incompl = cm[i][j] == bit[B] and not complete[components[i][j]]
        if unknown or black_incompl:
            ans = []

            # fill (i, j) with white
            cm_new = deepcopy(cm)
            cm_new[i][j] &= remove_except[W]
            if cm_new[i][j]:
                ans.append(cm_new)

            # fill (i, j) with black rectangle + white neighbourhood
            for m_ in divisors[x]:
                n_ = x // m_
                for k in range(x):
                    i0 = i - (k // n_)
                    j0 = j - (k % n_)
                    if not ((0 <= i0 <= m - m_) and (0 <= j0 <= n - n_)):
                        continue

                    i1, j1 = i0 + m_, j0 + n_
                    cm_new = deepcopy(cm)

                    # fill cm_new with a black rectangle of shape (m_, n_)
                    stopped = False
                    for i_, j_ in product(range(i0, i1), range(j0, j1)):
                        cm_new[i_][j_] &= remove_except[B]
                        if not cm_new[i_][j_]:
                            stopped = True
                            break
                    if stopped:
                        continue

                    # fill cm_new neighborhood of the rectangle with white
                    nbhd_u = [(i0 - 1, j_    ) for j_ in range(j0, j1)]
                    nbhd_d = [(i1    , j_    ) for j_ in range(j0, j1)]
                    nbhd_l = [(i_    , j0 - 1) for i_ in range(i0, i1)]
                    nbhd_r = [(i_    , j1    ) for i_ in range(i0, i1)]
                    nbhd = nbhd_u + nbhd_d + nbhd_l + nbhd_r

                    stopped = False
                    for i_, j_ in nbhd:
                        if not (0 <= i_ < m and 0 <= j_ < n):
                            continue
                        cm_new[i_][j_] &= remove_except[W]
                        if not cm_new[i_][j_]:
                            stopped = True
                            break
                    if stopped:
                        continue

                    ans.append(cm_new)

            return ans

    # find empty cell, fill with black/white
    func = lambda t: count[cm[t[0]][t[1]]] > 1
    i0, j0 = next(filter(func, product(range(m), range(n))))
    ans = []
    for x in range(b):
        cm_new = deepcopy(cm)
        cm_new[i0][j0] = 1 << x
        ans.append(cm_new)
    return ans


def prune(cm: Choices) -> Optional[Choices]:
    ''' Given partially filled grid, use constraints of the problem to
    remove numbers from the list of possibilities of each cell. '''
    cm = deepcopy(cm)
    edited = False

    # compute connected components
    components, num_components = ccs(cm)

    # areas[k] = area of cc with index k
    areas = [0 for _ in range(num_components)]
    for i, j in product(range(m), range(n)):
        areas[components[i][j]] += 1

    # colors[k] = color of cc with index k
    colors = [0 for _ in range(num_components)]
    for i, j in product(range(m), range(n)):
        colors[components[i][j]] = cm[i][j]

    # coordinates[k] = [(i, j) | components[i][j] = k]
    coordinates = [[] for _ in range(num_components)]
    for i, j in product(range(m), range(n)):
        coordinates[components[i][j]].append((i, j))

    neighborhood, complete = nbhds(cm, components, num_components)

    # complete and ((Black, not rectangle) or (White, rectangle)) -> reject
    for k in range(num_components):
        if count[colors[k]] == 1 and complete[k]:
            if value[colors[k]] == int(rectangle(coordinates[k])):
                return None

    # areas
    # num < area -> reject
    # num > area -> if complete, reject
    # num = area -> if not complete, surround with opposite color
    for k in range(num_components):
        if count[colors[k]] > 1:
            continue
        nums_k = set(grid[i][j] for i, j in coordinates[k] if grid[i][j])
        if len(nums_k) > 1:
            return None
        for num in nums_k:
            if num < areas[k]:
                return None
            elif num > areas[k]:
                if complete[k]:
                    return None
            else:
                if not complete[k]:
                    for i, j in neighborhood[k]:
                        cm[i][j] &= remove[value[colors[k]]]
                        if not cm[i][j]:
                            return None
                    edited = True

    # Two numbers orth adjacent and different => colors are different
    # (o = 0) => 2 cells are vertical, (o = 1) => 2 cells are horizontal
    # (u = 0) => [(i0,j0) | (i1,j1)], (u = 1) => [(i1,j1) | (i0,j0)]
    for o, u in product(range(2), repeat=2):
        ran_i = range(not o and u, m - (not o and not u))
        ran_j = range(u and o, u - (o and not u))
        for i0, j0 in product(ran_i, ran_j):
            i1 = i0 + (not o) * (-1 if u else 1)
            j1 = j0 + o * (-1 if u else 1)
            n0 = grid[i0][j0]
            n1 = grid[i1][j1]
            if n0 and n1 and n0 != n1:
                if count[cm[i0][j0]] == 1:
                    c = cm[i1][j1]
                    cm[i1][j1] &= remove[value[cm[i0][j0]]]
                    edited |= (cm[i1][j1] != c)
                    if not cm[i1][j1]:
                        return None

    # Diagonal is [black, black] => Opposite diagonal has equal values
    for i, j in product(range(m - 1), range(n - 1)):
        for o in range(4):
            i0, j0 = i + ((o + 0 - 1) % 4 < 2), j + ((o + 0) % 4 > 1)  # main
            i1, j1 = i + ((o + 1 - 1) % 4 < 2), j + ((o + 1) % 4 > 1)  # diag
            i2, j2 = i + ((o + 2 - 1) % 4 < 2), j + ((o + 2) % 4 > 1)  # op
            i3, j3 = i + ((o + 3 - 1) % 4 < 2), j + ((o + 3) % 4 > 1)  # diag
            if cm[i1][j1] == bit[B] and cm[i3][j3] == bit[B]:
                if count[cm[i2][j2]] == 1:
                    c = cm[i0][j0]
                    cm[i0][j0] &= remove_except[value[cm[i2][j2]]]
                    edited |= (cm[i0][j0] != c)
                    if not cm[i0][j0]:
                        return None

    return prune(cm) if edited else cm


def nbhds(cm: Grid, components: list[list[int]], num_components: int) -> tuple[
    list[list[tuple[int, int]]],
    list[int],
]:
    # neighborhood[k] = [(i,j) | (i,j) is orth adj to coordinates[k]]
    neighborhood = [[] for _ in range(num_components)]
    for o in range(2):
        for i0, j0 in product(range(m - (o == 0)), range(n - (o == 1))):
            i1 = i0 + (o == 0)
            j1 = j0 + (o == 1)
            k0 = components[i0][j0]
            k1 = components[i1][j1]
            if k0 != k1:
                neighborhood[k0].append((i1, j1))
                neighborhood[k1].append((i0, j0))

    # complete[k] = 1 if comp k is surrounded by opposite color, else 0
    complete = [0 for _ in range(num_components)]
    for k in range(num_components):
        complete[k] = int(all(count[cm[i][j]] == 1 for i, j in neighborhood[k]))

    return (neighborhood, complete)


def rectangle(arr: list[tuple[int, int]]) -> bool:
    ''' Given sorted list of coordinates, return True if and only if the
    coordinates are a rectangle. '''
    if not arr:
        return True
    i0, j0 = arr[0]
    i1, j1 = arr[-1][0] + 1, arr[-1][1] + 1
    m_, n_ = i1 - i0, j1 - j0
    if m_ * n_ != len(arr):
        return False
    for k, (i, j) in enumerate(arr):
        ik, jk = divmod(k, n_)
        if (i0 + ik, j0 + jk) != (i, j):
            return False
    return True


def ccs(xm: list[list[int]]) -> tuple[list[list[int]], int]:
    ''' Use a DFS to compute the connected components of the grid. '''
    direc = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    comps = [[0 for _ in range(n)] for _ in range(m)]
    n_ccs = 0

    for i, j in product(range(m), range(n)):
        if not comps[i][j]:
            n_ccs += 1
            # do a DFS starting from i, j, marking that cc with comps
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                comps[x][y] = n_ccs
                for dx, dy in direc:
                    x_ = x + dx
                    y_ = y + dy
                    if not (0 <= x_ < m and 0 <= y_ < n):
                        continue
                    if comps[x_][y_]:
                        continue
                    if xm[x][y] != xm[x_][y_]:
                        continue
                    stack.append((x_, y_))

    for i, j in product(range(m), range(n)):
        comps[i][j] -= 1

    return (comps, n_ccs)


# Solve the puzzle
# ----------------------------------------------------------------------

def printsol(xm: Grid) -> None:
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    xm = np.array(xm)
    gm = np.array(grid).astype('int').astype('str')
    gm[gm == '0'] = ''
    ax = sns.heatmap(
        xm,
        annot=gm,
        cbar=False,
        fmt='',
        linewidths=0.1,
        linecolor='black',
        square=True,
        cmap=['black', 'white']
    )
    ax.tick_params(
        left=False,
        labelleft=False,
        bottom=False,
        labelbottom=False
    )
    plt.show()


# 2 Possible choices:
# 1. Run script starting from scratch (~30 min on my machine)
starting_grid_slow = [[-1 for _ in range(n)] for _ in range(m)]

# 2. Fill grid by hand and use script to fill last cells (<1 min)
u = -1
starting_grid_fast = [
    [u, u, u, u, u, u, u, u, u, B, W, W, W, W, W, W, W, B, B, B],
    [u, u, u, u, u, u, u, u, u, W, B, B, B, B, B, B, W, B, B, B],
    [u, u, u, u, u, u, u, u, u, W, B, B, B, B, B, B, W, W, W, W],
    [u, u, u, u, u, u, u, u, W, B, W, W, W, W, W, W, B, B, B, B],
    [u, u, u, u, u, u, u, W, B, W, B, B, W, W, W, B, W, W, W, W],
    [u, W, u, u, W, W, W, B, W, W, B, B, W, B, W, B, W, B, B, B],
    [u, B, u, W, B, B, B, W, B, B, W, W, B, W, B, W, W, W, W, W],
    [u, B, u, W, B, B, B, W, B, B, W, W, B, W, B, W, B, B, B, B],
    [u, W, u, W, B, B, B, W, B, B, W, B, W, W, B, W, B, B, B, B],
    [W, W, W, B, W, W, W, W, W, W, B, W, u, u, W, B, W, W, W, W],
    [B, B, W, W, B, B, B, B, B, B, W, u, u, u, u, W, B, W, B, B],
    [B, B, W, W, B, B, B, B, B, B, W, u, u, u, u, B, W, W, B, B],
]

# Choose the solution with faster runtime
starting_grid = starting_grid_fast

t0 = time()
sol = solution(starting_grid)
ans = prod(sum(sol[i][j] for j in range(n)) for i in range(m))
t1 = time() - t0
print(f'Found solution in {t1 / 60:.2f} min.')
print(f'{ans = }')
print('sol = ')
pprint(sol)
printsol(sol)
