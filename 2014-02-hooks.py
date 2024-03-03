from z3 import Int, Solver, Or, Sum, sat
from itertools import product
from pprint import pprint

row_sum = [26, 42, 11, 22, 42, 36, 29, 32, 45]
col_sum = [31, 19, 45, 16,  5, 47, 28, 49, 45]

n = len(row_sum)
grid = list(product(range(n), repeat=2))
hook = [[(i, j) for i, j in grid if max(i, j) + 1 == k] for k in range(n+1)]

# Variables
x = [[Int(f'x{i}{j}') for j in range(n)] for i in range(n)]

# Solver and constraints
s = Solver()
s += [Or(x[i][j] == 0, x[i][j] == max(i, j) + 1) for i, j in grid]
s += [row_sum[i] == Sum([x[i][j] for j in range(n)]) for i in range(n)]
s += [col_sum[j] == Sum([x[i][j] for i in range(n)]) for j in range(n)]
s += [k**2 == Sum([x[i][j] for i, j in hook[k]]) for k in range(1, n+1)]

# Solution
if s.check() == sat:
    m = s.model()
    A = [[m.evaluate(x[i][j]).as_long() for j in range(n)] for i in range(n)]
    ans = sum([A[i][j] for i, j in grid if (i + j) % 2 == 0])
    print(f'{ans = }')
    print('grid = ')
    pprint(A)
else:
    print('no solution')
