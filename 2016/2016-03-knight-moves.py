from itertools import product

import numpy as np
from codetiming import Timer
from z3 import And, Implies, IntVector, PbEq, Solver, sat

# fmt: off
chessboard = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0,11, 0, 0, 0, 0],
    [0, 0, 0, 0, 0,14, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0,15, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int32)
# fmt: on

rows = np.array([10, 34, 108, 67, 63, 84, 24, 16], dtype=np.int32)
cols = np.array([7, 14, 72, 66, 102, 90, 42, 13], dtype=np.int32)
n = len(rows)

s = Solver()
X = np.array(IntVector("x", n**2)).reshape((n, n))

s += [And(x >= 0, x <= 28) for x in X.flat]
s += [x == y for x, y in zip(X.flat, chessboard.flat) if y]
s += [PbEq([(x == i, 1) for x in X.flat], 1) for i in range(1, 29)]
s += [s == r for s, r in zip(np.sum(X, axis=1), rows)]
s += [s == c for s, c in zip(np.sum(X, axis=0), cols)]
s += [Implies(x != 0, r != 0) for x, r in zip(X.flat, np.rot90(X).flat)]


directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
for i, j in product(range(n), repeat=2):
    knight_moves = [(i + di, j + dj) for di, dj in directions if 0 <= i + di < n and 0 <= j + dj < n]
    needs_move = And(X[i, j] != 0, X[i, j] != 28)
    moves = [(X[i, j] + 1 == X[k, l], 1) for k, l in knight_moves]
    s += Implies(needs_move, PbEq(moves, 1))


with Timer(initial_text="Checking z3 solver"):
    check = s.check()

if check == sat:
    m = s.model()
    A = np.vectorize(lambda x: m.evaluate(x).as_long())(X)

    B = np.concatenate([A, A.T], axis=0)
    C = np.where(B == 0, 1, B)
    ans = np.max(np.prod(C, axis=1))
    print(f"Answer: {ans}")
    print("Chessboard:\n", A, sep="")
# Checking z3 solver
# Elapsed time: 1.0316 seconds
# Answer: 19675656
# Chessboard:
# [[ 0  0 10  0  0  0  0  0]
#  [ 0  0  0 22  0 12  0  0]
#  [ 0  9 28 11 26 21  0 13]
#  [ 0  0  1  4 23 14 25  0]
#  [ 0  5  8 27 20  3  0  0]
#  [ 7  0 19  2 15 24 17  0]
#  [ 0  0  6  0 18  0  0  0]
#  [ 0  0  0  0  0 16  0  0]]
