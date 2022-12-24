# Imports
# ----------------------------------------------------------------------
from functools import partial
from itertools import product
from more_itertools import intersperse
from pysmt.typing import INT
from pysmt.shortcuts import Equals, Implies, Int, Not, Symbol, get_model, Plus
from collections import defaultdict
from typing import Optional


# Types
# ----------------------------------------------------------------------
Grid = list[list[Optional[int]]]
Point = tuple[int, int]
Island = int
Islands = dict[Island, tuple[int, int, Optional[int]]]
Bridge = tuple[Island, Island]
HashiSol = tuple[Island, dict[Bridge, int]]


# Functions
# ----------------------------------------------------------------------

def island_dict(grid: Grid) -> Islands:
    m, n = len(grid), len(grid[0])
    iter = product(range(m), range(n))
    func = lambda t: (x := t[0], y := t[1], grid[x][y])[-1]
    tupl = lambda t: (x := t[0], y := t[1], (x, y, grid[x][y]))[-1]
    return dict(enumerate(map(tupl, filter(func, iter))))


def valid(islands: Islands, i: Island, j: Island) -> bool:
    # i < j
    if i >= j:
        return False
    
    # i, j are aligned
    xi, yi, _ = islands[i]
    xj, yj, _ = islands[j]
    if not (xi == xj or yi == yj):
        return False

    # no island in the middle of i, j
    xx, XX = min(xi, xj), max(xi, xj)
    yy, YY = min(yi, yj), max(yi, yj)

    def obstructs(k):
        xk, yk, _ = islands[k]
        midx = (xk == xi and (yy < yk < YY))
        midy = (yk == yi and (xx < xk < XX))
        return midx or midy

    others = (k for k in islands.keys() if k != i and k != j)

    return not any(map(obstructs, others))


def solve(islands: Islands) -> Optional[HashiSol]:
    # Variables
    # ------------------------------------------------------------------
    is_valid = partial(valid, islands)
    n = len(islands)
    R = range(n)
    x = [[Symbol(f'x[{i},{j}]', INT) for j in R] for i in R]
    y = Symbol('y', INT)

    # Constraints
    # ------------------------------------------------------------------
    formula = True
    
    # bound for y
    formula &= (0 <= y) & (y < n)

    # bounds for x
    for i, j in product(R, R):
        if is_valid(i, j):
            formula &= (0 <= x[i][j]) & (x[i][j] <= 2)
        else:
            formula &= Equals(x[i][j], Int(0))

    # number of bridges
    for i in R:
        c0 = Not(Equals(y, Int(i)))
        c1 = Plus([Plus(x[i][j], x[j][i]) for j in R])
        c2 = Int(islands[i][2])
        formula &= Implies(c0, Equals(c1, c2))

    # bridges don't cross
    for i, j, k, l in product(R, R, R, R):
        if not (is_valid(i, j) and is_valid(k, l)): 
            continue
        if i == k and j == l:
            continue
        
        # check that the bridges cross
        xi, yi, _ = islands[i]
        xj, yj, _ = islands[j]
        xk, yk, _ = islands[k]
        xl, yl, _ = islands[l]
        vert_ij = xi == xj
        vert_kl = xk == xl

        if not (vert_ij ^ vert_kl):
            continue
        
        cx = min(xk, xl) < xi < max(xk, xl)
        cy = min(yi, yj) < yk < max(yi, yj)
        if vert_ij and not (cx and cy):
            continue

        cx = min(xi, xj) < xk < max(xi, xj)
        cy = min(yk, yl) < yi < max(yk, yl)
        if vert_kl and not (cx and cy):
            continue

        # if they cross and ij has bridge, kl can't have a bridge
        c0 = Not(Equals(x[i][j], Int(0)))
        c1 = Equals(x[k][l], Int(0))
        formula &= Implies(c0, c1)

    # Solution
    # ------------------------------------------------------------------
    model = get_model(formula)
    if not model:
        return None

    Y = model.get_value(y).constant_value()
    X = [[model.get_value(x[i][j]).constant_value() for j in R] for i in R]
    A = {(i, j): X[i][j] for i, j in product(R, R) if valid(islands, i, j)}
    return (Y, A)


def print_hashi(grid: Grid) -> str:
    m, n = len(grid), len(grid[0])
    islands = island_dict(grid)
    sol = solve(islands)

    if not sol:
        return 'Could not find solution.'

    Y, A = sol
    X = [[' ' for _ in range(n)] for _ in range(m)]

    for island in islands.values():
        x, y, num_bridges = island
        X[x][y] = str(num_bridges)

    for bridge in A.items():
        (i, j), bridges = bridge
        if not bridges:
            continue
        xi, yi, _ = islands[i]
        xj, yj, _ = islands[j]
        if xi == xj:
            for y in range(min(yi, yj) + 1, max(yi, yj)):
                X[xi][y] = '-' if bridges == 1 else '='
        else:
            for x in range(min(xi, xj) + 1, max(xi, xj)):
                X[x][yi] = '|' if bridges == 1 else '║'

    B = defaultdict(int, A.items())
    num_wrong = islands[Y][2] 
    num_right = sum(B[x, Y] + B[Y, x] for x in range(len(islands)))

    output0 = [
        f'\nSolution of Hashi puzzle',
        f'ind_wrong = {Y}',
        f'num_wrong = {num_wrong}',
        f'num_right = {num_right}',
    ]
    output1 = [''.join(line) for line in X]
    return '\n'.join(output0 + output1)


# Data
# ----------------------------------------------------------------------
n = None

grid_0 = [
    [4, n, n, n, n, n, 5, n, 3, n, n, n, n, n, 4, n, 1],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, 2, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [6, n, 2, n, n, n, 5, n, 5, n, n, n, 4, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, 3, n, n, n, n, n, n, n, n, n, n, n, 6, n, 4],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, 6, n, n, n, 3, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [2, n, n, n, n, n, n, n, 4, n, n, n, 4, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, 3, n, n, n, n, n, n, n, n, n, n, n, 5, n, 4],
]


grid_1 = [
    [4, n, 4, n, 4, n, n, n, n, n, 5, n, n, n, 2, n, 1],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, 4, n, n, n, n, n, n, n, n, n, 1, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [3, n, 2, n, 4, n, n, n, 4, n, 4, n, n, n, 5, n, 3],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [5, n, 4, n, 4, n, 3, n, n, n, 5, n, 1, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, 6, n, 4, n, 5, n, 4],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [6, n, 6, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [3, n, 4, n, 5, n, n, n, 2, n, 4, n, 2, n, 2, n, 4],
]


grid_2 = [
    [4, n, 4, n, 2, n, n, n, 2, n, 4, n, n, n, 4, n, 2],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, 4, n, n, n, n, n, n, n, 4, n, n, n, n, n, 1],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [4, n, 7, n, 6, n, 4, n, n, n, n, n, n, n, 5, n, 4],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, 5, n, n, n, n, n, n, n, n, n, 4, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, 4, n, 5, n, 7, n, 2, n, 2, n, 2],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [3, n, 4, n, n, n, n, n, n, n, 4, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [1, n, n, n, 3, n, n, n, n, n, 4, n, n, n, 4, n, 2],
]


grid_3 = [
    [4, n, 2, n, 3, n, 4, n, n, n, 5, n, 5, n, n, n, 4],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, 5, n, n, n, 6, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [4, n, 4, n, n, n, n, n, n, n, n, n, 6, n, n, n, 4],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, 4, n, n, n, 4, n, n, n, 4, n, 2],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [3, n, 4, n, 6, n, n, n, 4, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [n, n, n, n, n, n, n, n, n, n, 4, n, 4, n, 7, n, 3],
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n],
    [2, n, n, n, 3, n, n, n, 2, n, n, n, n, n, 4, n, 3],
]


# Solution
# ----------------------------------------------------------------------
print(print_hashi(grid_0))
print(print_hashi(grid_1))
print(print_hashi(grid_2))
print(print_hashi(grid_3))
ans = (8 * 9 * 10 * 12) / 13**4
print(f'\nans = {ans:.7f}')


'''
Solution of Hashi puzzle
ind_wrong = 7
num_wrong = 2
num_right = 5
4=====5-3-----4-1
║     ║ |     ║
║     ║ 2     ║
║     ║ |     ║
6=2===5 5===4 ║
║ |   | ║   ║ ║
║ 3   | ║   ║ 6=4
║ ║   | ║   ║ ║ ║
║ 6===3 ║   ║ ║ ║
║ ║     ║   ║ ║ ║
2 ║     4===4 ║ ║
  ║           ║ ║
  3-----------5=4

Solution of Hashi puzzle
ind_wrong = 20
num_wrong = 1
num_right = 4
4=4 4=====5---2 1       
║ ║ ║     ║   | |       
║ 4 ║     ║ 1 | |       
║ ║ ║     ║ | | |       
3 2 4===4=4 | 5=3       
|           | ║
5=4 4=3---5=1 ║
║ ║ ║     ║ | ║
║ ║ ║     6=4-5=4       
║ ║ ║     ║     ║       
6=6 ║     ║     ║       
║ ║ ║     ║     ║       
3-4-5===2 4=2 2=4       

Solution of Hashi puzzle
ind_wrong = 16
num_wrong = 5
num_right = 3
4=4 2   2 4===4-2       
║ ║ ║   ║ ║   | |       
║ 4 ║   ║ 4   | 1       
║ ║ ║   ║ ║   |
4=7-6=4 ║ ║   5=4       
  ║ | ║ ║ ║   ║ ║       
  ║ 5 ║ ║ ║   4 ║       
  ║ ║ ║ ║ ║   ║ ║       
  ║ ║ 4=5-7=2 2 2       
  ║ ║     ║
3=4 ║     4
|   ║     ║
1   3--1  4===4=2       

Solution of Hashi puzzle
ind_wrong = 21
num_wrong = 4
num_right = 1
4=2 3-4===5-5===4
║   ║ |   ║ ║   ║
║   ║ 5===6 ║   ║
║   ║ ║   ║ ║   ║
4=4 ║ ║   ║ 6===4
  ║ ║ ║   ║ ║
  ║ ║ 4===4 ║ 4=2
  ║ ║       ║ ║
3=4 6===4   ║ ║
|   ║   ║   ║ ║
|   ║   ║ 4-4-7=3
|   ║   ║     ║ |
2---3   2     4=3


Solution:
The bridges spell the words: 
    0. PROB 
    1. NO ACES
    2. GIVEN
    3. SHAPE

Shape = Right numbers = 5431

Question: what is P(bridge hand with shape 5431 has 0 aces)?

Ans: A shape of 5431 means that the hand has 
    - 5 cards of suit 0, 
    - 4 cards of suit 1, 
    - 3 cards of suit 2,
    - 1 cards of suit 3.

P(bridge hand with shape 5431 has 0 aces)
    =   P(draw 5 of suit 0 and none of them is an ace)
      * P(draw 4 of suit 1 and none of them is an ace)
      * P(draw 3 of suit 2 and none of them is an ace)
      * P(draw 1 of suit 3 and none of them is an ace)

P(draw k of suit j and none of them is an ace)
    = (12 * ... * (12 - (k - 1))) / (13 * ... * (13 - (k - 1)))
    = (12 - (k - 1)) / 13
    = (13 - k) / 13

Therefore,
Ans = P(bridge hand with shape 5431 has 0 aces)
    = ((13 - 5) * (13 - 4) * (13 - 3) * (13 - 1)) / 13^4
    = (8 * 9 * 10 * 12) / 13^4
    = 0.3025104
'''