from pprint import pprint
from z3 import Int, Solver, Distinct, sat, And, Or, Implies
from itertools import product

# Inputs
regs = [
    [ 0,  0, 22,  2,  2,  3,  3,  3,  4],
    [ 0,  0,  2,  2,  6,  7,  3,  3,  4],
    [ 0,  5,  6,  6,  6,  7,  7,  4,  4],
    [ 9,  5,  6, 21,  6, 12,  7, 10, 10],
    [ 9,  9, 13, 13, 14, 14, 16, 17, 17],
    [ 9,  9, 14, 14, 14, 15, 16, 18, 17],
    [19,  9, 23, 23, 20, 20, 18, 18, 18],
    [19,  8,  8, 11, 11, 20, 20,  1, 18],
    [19, 11, 11, 11, 11,  1,  1,  1, 18],
]

given = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

n = len(regs)
r = max(max(row) for row in regs) + 1

# coords[k] = [(i, j) | regs[i][j] == k]
coords = [[] for _ in range(r)]
for i, j in product(range(n), repeat=2):
    coords[regs[i][j]].append((i, j))

n_cells = [len(coords[k]) for k in range(r)]

# Define variables and solver
I = list(product(range(n), repeat=2))
X = [[Int(f'x{i}{j}') for j in range(n)] for i in range(n)]
s = Solver()

# Add constraints to the solver
s += [And(1 <= X[i][j], X[i][j] <= n_cells[regs[i][j]]) for i, j in I]
s += [X[i][j] == given[i][j] for i, j in I if given[i][j]]
s += [Distinct([X[i][j] for i, j in coords[k]]) for k in range(r)]

for i, j in I:
    for x in range(1, n_cells[regs[i][j]] + 1):
        drs = [(x, 0), (-x, 0), (0, x), (0, -x)]
        crd = [(i + a, j + b) for a, b in drs if 0 <= i + a < n and 0 <= j + b < n]
        col = [(i + a, j) for a in range(-x + 1, x) if a and 0 <= i + a < n]
        row = [(i, j + b) for b in range(-x + 1, x) if b and 0 <= j + b < n]
        
        c1 = Or([X[i_][j_] == x for i_, j_ in crd])
        c2 = And([X[i_][j_] != x for i_, j_ in row + col])
        s += Implies(X[i][j] == x, And(c1, c2))

# Solve the problem
if s.check() == sat:
    m = s.model()
    A = [[m.evaluate(X[i][j]).as_long() for j in range(9)] for i in range(9)]

    ans = 0
    for k in range(r):
        max_ = 0
        for i in range(n):
            arr = [A[i][j] for j in range(n) if regs[i][j] == k]
            num = sum(n * 10**e for e, n in enumerate(reversed(arr)))
            max_ = max(max_, num)
        ans += max_

    print(f'{ans = }')
    print('grid = ')
    pprint(A)
else:
    print('no solution')

'''
ans = 6647
grid =
    3 4 1 1 3 4 2 3 2
    2 5 2 4 2 3 5 1 3
    1 1 3 6 1 1 2 1 4
    3 2 4 1 5 1 4 1 2
    5 4 2 1 2 5 1 1 3
    1 2 3 4 1 1 2 3 2
    1 6 1 2 4 2 5 6 4
    3 2 1 3 1 1 3 1 1
    2 4 2 6 5 4 2 3 2
'''