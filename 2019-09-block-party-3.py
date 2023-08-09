from pprint import pprint
from z3 import Int, Solver, Distinct, sat, And, Or, Implies
from itertools import product

# Inputs
regs = [
    [ 0,  2,  2,  2,  3,  4,  5,  6,  7],
    [ 0,  1,  2,  3,  3,  3,  5,  7,  7],
    [ 0,  2,  2,  2,  3,  3,  5,  7,  7],
    [ 0,  0,  2,  8,  3,  9,  9,  7,  7],
    [10,  0, 11,  8, 12,  9, 13, 14, 14],
    [10, 10, 11, 11, 12,  9, 13, 14, 14],
    [10, 15, 16, 16, 12, 17, 17, 14, 14],
    [15, 15, 15, 16, 16, 16, 18, 19, 19],
    [15, 16, 16, 16, 18, 18, 18, 18, 19],
]

given = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0],
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
ans = 6092
grid = 
    4 7 8 1 1 1 3 1 7
    1 1 2 7 2 6 1 1 3
    5 3 6 4 3 5 2 4 6
    3 2 5 2 4 2 3 5 2
    4 6 3 1 1 4 2 6 3
    2 1 2 1 3 1 1 1 2
    3 1 1 5 2 1 2 4 5
    5 2 3 2 4 6 1 1 3
    4 3 8 7 3 4 2 5 2
'''