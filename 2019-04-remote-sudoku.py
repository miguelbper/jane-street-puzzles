from pprint import pprint
from z3 import Int, Solver, Distinct, sat, And, Or
from itertools import product

grid = [
    [2, 0, 0, 0, 7, 1, 8, 3, 6],
    [0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 5, 0, 0, 5, 4, 0, 2],
    [0, 0, 0, 1, 0, 0, 5, 0, 1],
    [8, 3, 3, 0, 1, 0, 2, 4, 4],
    [3, 0, 4, 0, 0, 3, 0, 0, 0],
    [6, 0, 2, 3, 0, 0, 5, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 0],
    [7, 2, 7, 3, 1, 0, 0, 0, 3],
]

# squares[k] = [(i, j) | (i, j) is in kth 3x3 square]
squares = [[] for _ in range(9)]
for k in range(9):
    i, j = divmod(k, 3)
    squares[k].extend(product(range(3*i, 3*i + 3), range(3*j, 3*j + 3)))

remote = []
for i, j in product(range(9), repeat=2):
    dist = grid[i][j]
    if dist:
        direc = [(dist, 0), (-dist, 0), (0, dist), (0, -dist)]
        neigh = [(i + di, j + dj) for di, dj in direc]
        coord = [(i, j) for i, j in neigh if 0 <= i < 9 and 0 <= j < 9]
        remote.append((dist, coord))

# Define variables and solver
X = [[Int(f'x{i}{j}') for j in range(9)] for i in range(9)]
s = Solver()

# Add constraints to the solver
s += [And(1 <= X[i][j], X[i][j] <= 9) for i in range(9) for j in range(9)]
s += [Distinct([X[i][j] for j in range(9)]) for i in range(9)]
s += [Distinct([X[i][j] for i in range(9)]) for j in range(9)]
s += [Distinct([X[i][j] for i, j in squares[k]]) for k in range(9)]
s += [Or([X[i][j] == dist for i, j in coords]) for dist, coords in remote]

# Solve the problem
if s.check() == sat:
    m = s.model()
    A = [[m.evaluate(X[i][j]).as_long() for j in range(9)] for i in range(9)]
    ans = sum([A[i][j]**2 for i in range(9) for j in range(9) if grid[i][j]])
    print(f'{ans = }')
    print('grid = ')
    pprint(A)
else:
    print('no solution')

'''
ans = 1105
grid =
6 9 2  5 3 7  1 8 4
1 3 7  8 4 2  5 6 9
5 8 4  1 9 6  2 3 7

4 5 3  7 8 9  6 1 2
9 1 6  4 2 3  7 5 8
7 2 8  6 1 5  4 9 3

2 7 1  9 5 8  3 4 6
8 6 5  3 7 4  9 2 1
3 4 9  2 6 1  8 7 5
'''
