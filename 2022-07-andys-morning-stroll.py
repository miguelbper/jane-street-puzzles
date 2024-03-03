import numpy as np
from sympy import Matrix, eye, zeros, shape, Rational
from functools import cache
from itertools import product

# ======================================================================
# 1. footbal
# ======================================================================

# solution using intuitive argument
# ----------------------------------------------------------------------

'''
Solution: expected number of steps = 20
Proof: Imagine a random walk of the ant, which continues forever. There
are 20 hexagons. By symmetry, 1/20 of the time the ant is in its
starting hexagon. The average time between two consecutive occurences of
the ant being in the starting hexagon is 20.
'''


# solution using Markov chains
# ----------------------------------------------------------------------

# States = {0, 1, 2, 3, 4, 5}. Each state is the distance from home.
P = Matrix([[0, 3, 0, 0, 0, 0],
            [1, 0, 2, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 2, 0, 1],
            [0, 0, 0, 0, 3, 0]]) / 3

# compute fixed row vector w, which satisfies wP = w and sum(w) = 1
w = (P - eye(6)).T.nullspace()[0]
w = w / sum(w)

# Theorem: E(return time for state i) = 1 / w[i]
r = 1 / w[0]

print(f'expected number of steps = {r}')
# expected number of steps = 20


# ======================================================================
# 2. kitchen
# ======================================================================


# solution using recursion
# ----------------------------------------------------------------------

'''
Consider the following coordinates (i, j)

       (-1, 1)   (-1, 0)        B         (-1, -1)
(0, 1)         B         (0, 0)   (0, -1)
       ( 1, 1)   ( 1, 0)        B         ( 1, -1)
(2, 1)         B         (2, 0)   (2, -1)
'''


@cache
def prob(i, j, n):
    '''
    probability that the ant can reach (0, 0) starting at (i ,j)
    in n steps (remaining, out of 20 total steps)
    '''
    if (i, j) == (0, 0) and n < 20:
        return 1
    if (i, j) != (0, 0) and n == 0:
        return 0

    p1 = prob(i - 1, j, n - 1)              # left
    p2 = prob(i + 1, j, n - 1)              # right
    p3 = prob(i, j - (-1)**(i + j), n - 1)  # up / bottom

    return (p1 + p2 + p3) / 3


print(f'p = {1 - prob(0, 0, 20):.7f} (computed using recursion)')
# p = 0.4480326


# solution using Markov chains and symmetry to simplify the state space
# ----------------------------------------------------------------------

'''
By symmetry, it is enough to consider only the following states, where
00 is the home state:

                00
              01  BB
            BB  02  03
          04  05  0B  06
        07  BB  08  09  0B
      BB  10  11  0B  12  13
    14  15  BB  16  17  BB  18
  19  BB  20  21  BB  22  23  BB
BB  24  25  BB  26  27  BB  28  29
  30  BB  31  32  BB  33  34
        35  BB  36  37
              38

If the ant were to leave the left/right boundary of this cone, then it
is 'reflected' back inside (symmetry).
'''

# a[state i] = three neighbours of i (with repeats for reflections)
# we make 0 into an absorbing state, but we will assume a start at 1
a = {
     0: ( 0,  0,  0),
     1: ( 0,  2,  2),
     2: ( 1,  3,  5),
     3: ( 2,  2,  6),
     4: ( 5,  5,  7),
     5: ( 2,  4,  8),
     6: ( 3,  9,  9),
     7: ( 4, 10, 10),
     8: ( 5,  9, 11),
     9: ( 6,  8, 12),
    10: ( 7, 11, 15),
    11: ( 8, 10, 16),
    12: ( 9, 13, 17),
    13: (12, 12, 18),
    14: (15, 15, 19),
    15: (10, 14, 20),
    16: (11, 17, 21),
    17: (12, 16, 22),
    18: (13, 23, 23),
    19: (14, 24, 24),
    20: (15, 21, 25),
    21: (16, 20, 26),
    22: (17, 23, 27),
    23: (18, 22, 28),
    24: (19, 25, 30),
    25: (20, 24, 31),
    26: (21, 27, 32),
    27: (22, 26, 33),
    28: (23, 29, 34),
    29: (29, 29, 29),
    30: (30, 30, 30),
    31: (25, 32, 35),
    32: (26, 31, 36),
    33: (27, 34, 37),
    34: (34, 34, 34),
    35: (35, 35, 35),
    36: (32, 37, 38),
    37: (37, 37, 37),
    38: (38, 38, 38)
}

# define transition matrix
P = zeros(39)
for i, j in product(range(39), range(3)):
    P[i, a[i][j]] += 1
P /= 3

# compute probability
p = 1 - (P ** 19)[1, 0]

print(f'p = {float(p):.7f} (computed using Markov chains and symmetry)')
# p = 0.4480326


# solution using Markov chains without simplifying state space
# ----------------------------------------------------------------------

# functions that define a hexagonal grid
def hexagonal_grid(n):
    if n == 1:
        return np.array([[0, 0, -1], [-1, 0, 0]])
    else:
        grid = hexagonal_grid(n-1)
        row = np.concatenate((grid, grid), axis=1)
        col = np.concatenate((row, row), axis=0)
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
        return (0, 0)
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
    coords = [(i - 1, j            ),
              (i - 1, j - (-1)**(i)),
              (    i, j - 1        ),
              (    i, j + 1        ),
              (i + 1, j            ),
              (i + 1, j - (-1)**(i))]
    coords = [(a, b) for (a, b) in coords if a in range(x) and b in range(y)]
    return coords


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
    last_state = grid[-1, -1]
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
    last_state = grid[-1, -1]
    markov_temp = markov(n)
    markov_temp[initial_state, :] = eye(last_state + 1)[initial_state, :]
    return markov_temp


# compute result
print('computing with Markov chains (no symmetry)...', end='\r')
n = 5
(i, j) = starting_coordinates(n)
start = hexagonal_grid_numbered(n)[i, j]
neighbour = neighbour_states(n, i, j)[0]
P = markov_absorbing(n)
A = P**19
p = 1 - A[neighbour, start]
print(f'p = {float(p):.7f} (computed using Markov chains, no symmetry)')
# p = 0.4480326
