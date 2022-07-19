from sympy import *
import numpy as np
from itertools import product

# -----------
# 1. football
# -----------

P = Matrix([[0, 3, 0, 0, 0, 0],
            [1, 0, 2, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 2, 0, 1],
            [0, 0, 0, 0, 3, 0]]) / 3
w = (P - eye(6)).T.nullspace()[0]
w = w / sum(w)
r = 1 / w[0]
print(f'1. expected number of steps = {r}')
# expected number of steps = 20

# ----------
# 2. kitchen
# ----------

# -------------------------------------
# method 1. define markov chain by hand

a = { 
     0: ( 0, 0, 0),
     1: ( 0, 2, 2),
     2: ( 1, 3, 5),
     3: ( 2, 2, 6),
     4: ( 5, 5, 7),
     5: ( 2, 4, 8),
     6: ( 3, 9, 9),
     7: ( 4,10,10),
     8: ( 5, 9,11),
     9: ( 6, 8,12),
    10: ( 7,11,15),
    11: ( 8,10,16),
    12: ( 9,13,17),
    13: (12,12,18),
    14: (15,15,19),
    15: (10,14,20),
    16: (11,17,21),
    17: (12,16,22),
    18: (13,23,23),
    19: (14,24,24),
    20: (15,21,25),
    21: (16,20,26),
    22: (17,23,27),
    23: (18,22,28),
    24: (19,25,30),
    25: (20,24,31),
    26: (21,27,32),
    27: (22,26,33),
    28: (23,29,34),
    29: (29,29,29),
    30: (30,30,30),
    31: (25,32,35),
    32: (26,31,36),
    33: (27,34,37),
    34: (34,34,34),
    35: (35,35,35),
    36: (32,37,38),
    37: (37,37,37),
    38: (38,38,38)
}

P = zeros(39)
for i in range(39):
    for j in range(3):
        P[i, a[i][j]] += 1
P /= 3

p = 1 - (P ** (floor(r) - 1))[1, 0]

print('2. probability')
print(f'\tmethod 1: p = {p} = {float(p):.7f}')
# p = 173576992/387420489 = 0.4480326


# -------------------------------------
# method 2. define grid programatically

# functions that define a hexagonal grid
def hexagonal_grid(n):
    if n == 1:
        return np.array([[0,0,-1], [-1,0,0]])
    else:
        grid = hexagonal_grid(n-1)
        row = np.concatenate((grid, grid), axis = 1)
        col = np.concatenate((row, row), axis = 0)
        return col


def hexagonal_grid_numbered(n):
    grid = hexagonal_grid(n)
    x, y = shape(grid)
    count = 0
    for i, j in product(range(x), range(y)):
        if grid[i, j] == 0:
            grid[i, j] = count
            count += 1
    return grid


# helper functions
def starting_coordinates(n):
    if n == 1:
        return (0,0)
    else:
        i = 2**(n - 1)
        j = 3 * 2**(n - 2)
        return (i, j)


def grid_shape(n):
    x = 2**n
    y = 3 * 2**(n - 1)
    return (x, y)


def neighbour_coordinates(n, i, j):
    x, y = grid_shape(n)
    coordinates = [(i - 1, j            ),
                   (i - 1, j - (-1)**(i)),
                   (    i, j - 1        ),
                   (    i, j + 1        ),
                   (i + 1, j            ),
                   (i + 1, j - (-1)**(i))]
    coordinates = [(a, b) for (a, b) in coordinates if a in range(x) and b in range(y)]
    return coordinates


def neighbour_states(n, i, j):
    grid = hexagonal_grid_numbered(n)
    coordinates = neighbour_coordinates(n, i, j)
    neighbours = [grid[a, b] for (a, b) in coordinates]
    neighbours = [n for n in neighbours if n != -1]
    return neighbours


# functions that define the markov transition matrix
def markov(n):
    grid = hexagonal_grid_numbered(n)
    x, y = shape(grid)
    last_state = grid[-1,-1]
    m = last_state + 1
    P = zeros(m)
    for i, j in product(range(x), range(y)):
        state = grid[i, j]
        if state != -1:
            neighbours = neighbour_states(n, i, j)
            n_neighbours = len(neighbours)
            for neighbour in neighbours:
                P[state, neighbour] += Rational(1, n_neighbours)
    return P


def markov_absorbing(n):
    (x, y) = starting_coordinates(n)
    grid = hexagonal_grid_numbered(n)
    initial_state = grid[x, y]
    last_state = grid[-1,-1]
    markov_temp = markov(n)
    markov_temp[initial_state, :] = eye(last_state + 1)[initial_state, :]
    return markov_temp


# compute result
print('\tmethod 2:')
for n in range(1,6):
    (i, j) = starting_coordinates(n)
    grid = hexagonal_grid_numbered(n)
    start = grid[i, j]
    neighbours = neighbour_states(n, i, j)
    P = markov_absorbing(n)
    A = P**(floor(r) - 1)
    print(f'\n\tn = {n}, (x,y) = {grid_shape(n)}, n_states = {grid[-1,-1] + 1}, start = {start}, neighbours = {neighbours}')
    for neighbour in neighbours:
        p = 1 - A[neighbour, start]
        print(f'\tneighbour = {neighbour}, p = {p} = {float(p):.7f}')
# p = 173576992/387420489 = 0.4480326


# helper function to print the grid
def print_grid(n):
    (i, j) = starting_coordinates(n)
    grid = hexagonal_grid_numbered(n)
    start_state = grid[i, j]
    print(f'\nwill print grid for n = {n}')
    print(f'(i,j) = {(i,j)}')
    print(f'start_state = {start_state}')

    indent = False
    for row in grid:
        row_string = [f'{x:02d}' if x >= 0 else 'BB' for x in row]
        row_string = '  '.join(row_string)
        if indent:
            row_string = '  ' + row_string
        print(row_string)
        indent = not indent