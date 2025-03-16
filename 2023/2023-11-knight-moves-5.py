from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sympy import Matrix

# Z = Matrix([
#     [11, 10, 11, 14], # 4
#     [ 8,  6,  9,  9], # 3
#     [10,  4,  3,  1], # 2
#     [ 7,  6,  5,  0], # 1
#     # a   b   c   d
# ])

# fmt: off
Z = Matrix([
    [9, 8 , 10, 12, 11, 8 , 10, 17],  # 8
    [7, 9 , 11, 9 , 10, 12, 14, 12],  # 7
    [4, 7 , 5 , 8 , 8 , 6 , 13, 10],  # 6
    [4, 10, 7 , 9 , 6 , 8 , 7 , 9 ],  # 5
    [2, 6 , 4 , 2 , 5 , 9 , 8 , 11],  # 4
    [0, 3 , 1 , 4 , 2 , 7 , 10, 7 ],  # 3
    [1, 2 , 0 , 1 , 2 , 5 , 7 , 6 ],  # 2
    [0, 2 , 4 , 3 , 5 , 6 , 2 , 4 ],  # 1
    #a  b   c   d   e   f   g   h
])
# fmt: on
m = Z.shape[0]
V = Matrix([[int((i, j) == (m - 1, 0)) for j in range(m)] for i in range(m)])
xy = "a1"
initial_state = (Z, V, xy, 0)


# types & functions
# ------------------------------------------------------------------------------

Time = int
Coord = str
Move = tuple[Time, Coord]
State = tuple[Matrix, Matrix, Coord, Time]


def to_coord(t: tuple[int, int]) -> Coord:
    x, y = t
    c = chr(ord("a") + y)
    i = m - x
    return f"{c}{i}"


def to_tuple(crd: Coord) -> tuple[int, int]:
    x = m - int(crd[1])
    y = ord(crd[0]) - ord("a")
    return (x, y)


def jump(state: State, move: Move) -> State | None:
    """Given current state and a move, compute the next state.

    If the move is not legal, return None.
    """
    # parse state
    Z, V, xy, t = state
    x, y = to_tuple(xy)

    # parse move
    dt, xy_ = move
    x_, y_ = to_tuple(xy_)
    t_ = t + dt

    # wait dt units
    def dZ_entry(i: int, j: int) -> int:
        c0 = -int(Z[i, j] == Z[x, y])
        c1 = int((i, j) == (m - 1 - x, m - 1 - y))
        return c0 + c1

    dZ = Matrix([[dZ_entry(i, j) for j in range(m)] for i in range(m)])
    n = sum(dZ[i, j] == -1 for i in range(m) for j in range(m))
    Z_ = Z + dZ * dt / n

    # check if move is legal
    dx = abs(x_ - x)
    dy = abs(y_ - y)
    dz = abs(Z_[x_, y_] - Z_[x, y])
    if {dx, dy, dz} != {0, 1, 2} or V[x_, y_] >= 3:
        return None

    V_ = deepcopy(V)
    V_[x_, y_] += 1
    return (Z_, V_, xy_, t_)


def verify(initial_state: State, moves: list[Move]) -> bool:
    # simulate all jumps
    state = initial_state
    for move in moves:
        state = jump(state, move)
        if not state:
            return False

    # get final state
    Z, V, xy, t = state
    x, y = to_tuple(xy)
    x_, y_ = 0, m - 1

    # do some checks
    # jump is between points of differing altitudes: check manually
    if (x, y) != (x_, y_):
        return False
    if V[x_, y_] > 1:
        return False
    if any(V[i, j] > 3 for i in range(m) for j in range(m)):
        return False
    return t >= 180


# solution & verification
# ------------------------------------------------------------------------------

# moves = [
#     (1, 'b3'), (0, 'a3'), (0, 'a2'), (0, 'b4'), (5, 'a2'),
#     (3, 'c1'), (2, 'a2'), (0, 'c1'), (15, 'd1'), (1, 'd2'),
#     (0, 'c2'), (1, 'b2'), (0, 'b1'), (0, 'b3'), (0, 'c3'),
#     (0, 'c4'), (0, 'a4'), (0, 'b4'), (0, 'd4')
# ]

moves = [
    (0, "b1"),
    (0, "c1"),
    (36, "c2"),
    (0, "a1"),
    (0, "b1"),
    (42, "c1"),
    (0, "c2"),
    (0, "a1"),
    (3, "c2"),
    (0, "d2"),
    (0, "d1"),
    (0, "e1"),
    (4, "f1"),
    (0, "f3"),
    (0, "f5"),
    (0, "h5"),
    (0, "h4"),
    (18, "h3"),
    (0, "g5"),
    (0, "e5"),
    (0, "e6"),
    (0, "e7"),
    (0, "f7"),
    (0, "f8"),
    (0, "d8"),
    (12, "b8"),
    (0, "c8"),
    (0, "e7"),
    (0, "e6"),
    (0, "g6"),
    (0, "g4"),
    (0, "g2"),
    (45, "f2"),
    (0, "e4"),
    (0, "e5"),
    (0, "f5"),
    (0, "d5"),
    (25, "e5"),
    (0, "e6"),
    (0, "e7"),
    (0, "f7"),
    (0, "f8"),
    (0, "g8"),
    (0, "h8"),
]

print(f"valid_solution = {verify(initial_state, moves)}")
print("sol = ", end="")
for t, xy in moves:
    print(f"({t}, {xy})", end=", ")


# drawings
# ------------------------------------------------------------------------------
states = [initial_state]
for move in moves:
    states.append(jump(states[-1], move))

nn = 10
mm = len(moves) // nn + 1
fs = 0.5
fig, axes = plt.subplots(mm, nn, figsize=(fs * mm, fs * nn))
cmap = ListedColormap(["white", "yellow", "green"])

for k, state in enumerate(states):
    i, j = divmod(k, nn)
    ax = axes[i, j]
    Z, V, xy, t = state
    x, y = to_tuple(xy)
    A = np.array(Z).astype(np.float32)
    C = np.zeros((m, m))
    C[0, m - 1] = 2
    C[x, y] = 1
    sns.heatmap(C, annot=A, cmap=cmap, cbar=False, ax=ax, vmin=0, vmax=2, linewidths=0.5, linecolor="gray")
    caption = f"({moves[k - 1][0]}, {moves[k - 1][1]})" if k else "initial"
    ax.set_title(caption)
    ax.set_xticks([])
    ax.set_yticks([])
for k in range(len(states), mm * nn):
    i, j = divmod(k, nn)
    ax = axes[i, j]
    ax.set_visible(False)
plt.show()
"""Sol = (0, b1), (0, c1), (36, c2), (0, a1), (0, b1), (42, c1), (0, c2), (0,
a1), (3, c2), (0, d2), (0, d1), (0, e1), (4, f1), (0, f3), (0, f5), (0, h5),
(0, h4), (18, h3), (0, g5), (0, e5), (0, e6), (0, e7), (0, f7), (0, f8), (0,
d8), (12, b8), (0, c8), (0, e7), (0, e6), (0, g6), (0, g4), (0, g2), (45, f2),
(0, e4), (0, e5), (0, f5), (0, d5), (25, e5), (0, e6), (0, e7), (0, f7), (0,
f8), (0, g8), (0, h8)"""
