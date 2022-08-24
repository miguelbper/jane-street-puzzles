# ======================================================================
# imports and definitions
# ======================================================================

import numpy as np
from itertools import product
from pysmt.typing import INT
from pysmt.shortcuts import (And, Equals, Implies, Int, Not, Or, Symbol, 
    get_model)

# define grids
n = None
values = [[n, 3, n, n, n, 7, n, n, n, n],
          [n, n, n, 4, n, n, n, n, n, n],
          [n, n, n, n, n, n, n, n, 2, n],
          [n, n, n, 1, n, n, n, n, n, n],
          [6, n, 1, n, n, n, n, n, n, n],
          [n, n, n, n, n, n, n, 3, n, 6],
          [n, n, n, n, n, n, 2, n, n, n],
          [n, 2, n, n, n, n, n, n, n, n],
          [n, n, n, n, n, n, 6, n, n, n],
          [n, n, n, n, 5, n, n, n, 2, n]]

region = [[ 0,  1,  1,  1,  2,  2,  2,  2,  2,  2],
          [ 0,  0,  1,  1,  1,  2,  3,  3,  2,  2],
          [ 0,  0,  4,  4,  5,  5,  6,  3,  7,  7],
          [ 0,  0,  8,  4,  9,  6,  6,  6,  7,  7],
          [ 0,  8,  8,  9,  9, 10, 11,  6,  6,  7],
          [ 0, 12,  8, 13, 14, 10, 15, 15,  7,  7],
          [ 0, 16, 17, 13, 13, 13, 15, 20, 20, 20],
          [16, 16, 17, 18, 13, 19, 22, 21, 21, 22],
          [16, 16, 16, 18, 18, 22, 22, 21, 21, 22],
          [16, 16, 16, 16, 18, 18, 22, 22, 22, 22]]

R = range(10)

# helper functions
def r(i, j):
    ''' region i, j belongs to '''
    return region[i][j]

def s(r):
    ''' number of elements in region k '''
    return len([None for i, j in product(R, R) if region[i][j] == r])

def dist(i, j, i_, j_):
    return abs(i - i_) + abs(j - j_)

# ======================================================================
# initialize variables
# ======================================================================

x = [[Symbol(f'x{i}{j}', INT) for j in R] for i in R]


# ======================================================================
# constraints
# ======================================================================


# initial given values
givens = And([
    Equals(x[i][j], Int(values[i][j])) 
    for i, j in product(R, R) 
    if values[i][j] != None
])


# {x | x is in region k} < {1,...,N}.
bounds = And([(1 <= x[i][j]) & (x[i][j] <= s(r(i, j))) 
              for i, j in product(R, R)])


# if x, y are in region k, in different cells then x != y
distinct = True
for (i, j), (i_, j_) in product(product(R, R), product(R, R)):
    if r(i, j) == r(i_, j_) and (i, j) != (i_, j_):
        distinct &= Not(Equals(x[i][j], x[i_][j_]))


# nearest (wrt taxicab) k to a k is at distance k
distances = True
for (i, j) in product(R, R):
    for k in range(1, s(r(i, j)) + 1):
        le_k = []
        eq_k = []
        for (i_, j_) in product(R, R):
            if abs(i - i_) + abs(j - j_) < k and (i, j) != (i_, j_):
                le_k += [Not(Equals(x[i][j], x[i_][j_]))]
            if abs(i - i_) + abs(j - j_) == k:
                eq_k += [Equals(x[i][j], x[i_][j_])]
        distances &= Implies(Equals(x[i][j], Int(k)), And(And(le_k), Or(eq_k)))


# ======================================================================
# solve
# ======================================================================

formula = givens & bounds & distinct & distances
model = get_model(formula)
A = [[model.get_value(x[i][j]).constant_value() for j in R] for i in R]

board = np.array(A)
solution = sum(np.prod(board, axis = 1))
print(f'sum of products = {solution},\nboard = \n{board}')