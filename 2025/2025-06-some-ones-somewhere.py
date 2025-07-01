from itertools import chain, groupby
from operator import itemgetter
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from codetiming import Timer
from ortools.sat.python import cp_model

Square = tuple[int, int, int]
Cell = tuple[int, int]
Board = list[Square | None]
Solution = list[Square]

T: int = 45
square_sizes: list[int] = [i for i in range(1, 10) for _ in range(i)]

# Define color scheme based on square size
size_colors: dict[int, str] = {
    2: "#006041",  # green
    3: "#f68e00",  # orange
    4: "#00328b",  # navy
    5: "#a3003c",  # burgundy
    6: "#0092d0",  # picton blue
    7: "#f4c500",  # yellow
    8: "#6e4f35",  # brown
    9: "#d2d6cd",  # celeste
}


squares_00: list[Square] = [
    (9 , 0 , 9),
    (18, 0 , 9),
    (27, 0 , 9),
    (36, 0 , 9),
    (37, 9 , 8),
    (29, 9 , 8),
    (24, 9 , 5),
    (19, 9 , 5),
    (13, 9 , 6),
    (38, 38, 7),
    (38, 31, 7),
    (30, 37, 8),
    (22, 37, 8),
    (30, 29, 8),
    (15, 38, 7),
    (8 , 38, 7),
    (0 , 37, 8),
    (0 , 33, 4),
    (4 , 33, 4),
    (0 , 24, 9),
    (0 , 15, 9),
    (0 , 11, 4),
    (4 , 12, 3),
    (8 , 33, 5),
    (13, 29, 9),
]  # fmt: skip

squares_01: list[Square] = [
    (0 , 0 , 4),
    (0 , 4 , 4),
    (0 , 8 , 4),
    (0 , 12, 7),
    (0 , 19, 7),
    (0 , 26, 6),
    (0 , 32, 8),
    (0 , 40, 5),
    (5 , 40, 5),
    (4 , 0 , 9),
    (4 , 9 , 3),
    (13, 0 , 8),
    (13, 8 , 8),
    (10, 36, 9),
    (19, 36, 9),
    (19, 27, 9),
    (36, 0 , 9),
    (36, 9 , 9),
    (36, 18, 9),
    (36, 27, 9),
    (36, 36, 9),
    (29, 0 , 7),
    (29, 7 , 7),
    (29, 14, 7),
    (28, 21, 8),
    (28, 29, 8),
    (28, 37, 8),
    (25, 21, 3),
    (24, 16, 5),
]  # fmt: skip

squares_02: list[Square] = [
    (0 , 0 , 9),
    (9 , 0 , 9),
    (18, 0 , 9),
    (27, 0 , 9),
    (36, 0 , 9),
    (0 , 9 , 8),
    (8 , 9 , 8),
    (16, 9 , 7),
    (0 , 17, 9),
    (9 , 17, 7),
    (0 , 26, 5),
    (5 , 26, 4),
    (5 , 30, 2),
    (0 , 36, 9),
    (38, 9 , 7),
    (39, 16, 6),
    (39, 22, 6),
    (36, 28, 9),
    (27, 28, 9),
    (32, 16, 7),
    (34, 23, 5),
    (29, 23, 5),
    (29, 20, 3),
    (24, 28, 3),
    (21, 31, 6),
    (15, 31, 6),
    (41, 41, 4),
    (41, 37, 4),
    (33, 37, 8),
    (25, 37, 8),
    (17, 37, 8),
]  # fmt: skip

squares_10: list[Square] = [
    (0 , 0 , 9),
    (0 , 9 , 9),
    (9 , 0 , 8),
    (17, 0 , 5),
    (22, 0 , 5),
    (27, 0 , 9),
    (36, 0 , 9),
    (36, 9 , 9),
    (36, 18, 9),
    (17, 5 , 3),
    (16, 8 , 4),
    (9 , 8 , 7),
    (9 , 15, 3),
    (0 , 38, 7),
    (0 , 31, 7),
    (20, 5 , 7),
    (0 , 18, 6),
    (6 , 18, 6),
    (7 , 37, 8),
    (15, 37, 8),
    (23, 37, 8),
    (31, 37, 8),
    (39, 39, 6),
    (39, 33, 6),
    (35, 33, 4),
    (33, 35, 2),
    (28, 32, 5),
    (15, 33, 4),
    (19, 28, 9),
]  # fmt: skip

squares_11: list[Square] = [
    (36, 36, 9),
    (0 , 36, 9),
    (0 , 29, 7),
    (0 , 22, 7),
    (7 , 31, 5),
    (2 , 0 , 4),
    (6 , 0 , 7),
    (13, 0 , 7),
    (20, 0 , 8),
    (20, 8 , 8),
    (28, 0 , 8),
    (28, 8 , 8),
    (28, 16, 8),
    (36, 0 , 9),
    (36, 9 , 9),
    (29, 38, 7),
    (29, 31, 7),
    (21, 37, 8),
    (25, 33, 4),
    (15, 39, 6),
    (12, 42, 3),
    (6 , 7 , 5),
    (6 , 12, 5),
    (11, 7 , 9),
]  # fmt: skip

squares_12: list[Square] = [
    (0 , 0 , 9),
    (9 , 0 , 9),
    (18, 0 , 9),
    (27, 0 , 9),
    (36, 0 , 9),
    (0 , 9 , 4),
    (0 , 13, 4),
    (0 , 17, 7),
    (0 , 24, 7),
    (0 , 31, 7),
    (0 , 38, 7),
    (7 , 25, 8),
    (7 , 33, 6),
    (7 , 39, 6),
    (13, 36, 9),
    (37, 9 , 8),
    (29, 9 , 8),
    (26, 9 , 3),
    (29, 17, 7),
    (36, 17, 9),
    (40, 40, 5),
    (40, 35, 5),
    (34, 39, 6),
    (28, 39, 6),
    (22, 39, 6),
    (22, 32, 7),
    (20, 34, 2),
]  # fmt: skip

squares_20: list[Square] = [
    (0 , 0 , 9),
    (9 , 0 , 9),
    (18, 0 , 9),
    (27, 0 , 9),
    (36, 0 , 9),
    (0 , 9 , 7),
    (7 , 9 , 8),
    (15, 9 , 7),
    (15, 16, 7),
    (12, 17, 3),
    (22, 9 , 7),
    (29, 9 , 8),
    (37, 9 , 8),
    (36, 17, 9),
    (36, 26, 4),
    (40, 26, 5),
    (40, 31, 5),
    (36, 36, 9),
    (38, 34, 2),
    (28, 37, 8),
    (20, 37, 8),
    (16, 41, 4),
    (0 , 37, 8),
    (0 , 28, 9),
    (0 , 22, 6),
]  # fmt: skip

squares_21: list[Square] = [
    (0 , 0 , 9),
    (0 , 9 , 9),
    (0 , 18, 9),
    (0 , 27, 4),
    (0 , 31, 5),
    (0 , 36, 9),
    (9 , 38, 7),
    (16, 38, 7),
    (23, 37, 8),
    (5 , 34, 2),
    (9 , 0 , 9),
    (18, 0 , 9),
    (27, 0 , 9),
    (34, 42, 3),
    (37, 37, 8),
    (37, 29, 8),
    (41, 25, 4),
    (9 , 9 , 7),
    (16, 9 , 7),
    (9 , 16, 6),
    (15, 16, 6),
    (21, 16, 8),
    (9 , 22, 5),
    (14, 22, 7),
]  # fmt: skip


squares_22: list[Square] = [
    (0 , 0 , 6),
    (0 , 6 , 6),
    (0 , 12, 6),
    (0 , 18, 5),
    (0 , 23, 4),
    (6 , 0 , 8),
    (6 , 8 , 8),
    (36, 0 , 9),
    (36, 9 , 9),
    (36, 18, 9),
    (36, 27, 9),
    (36, 36, 9),
    (27, 36, 9),
    (18, 36, 9),
    (29, 0 , 7),
    (22, 0 , 7),
    (28, 7 , 8),
    (28, 15, 8),
    (28, 23, 8),
    (31, 31, 5),
    (26, 31, 5),
    (21, 31, 5),
    (21, 24, 7),
    (24, 20, 4),
    (25, 7 , 3),
]  # fmt: skip


def sort_and_pad(squares: list[Square]) -> list[Square | None]:
    def pad_group(size: int, group: list[Any]) -> list[Any | None]:
        return group + [None] * (size - len(group))

    sorted_squares = sorted(squares, key=itemgetter(2))
    groups = {k: list(v) for k, v in groupby(sorted_squares, key=itemgetter(2))}
    groups_all = {k: groups.get(k, []) for k in range(1, 10)}
    return list(chain.from_iterable(pad_group(size, group) for size, group in groups_all.items()))


boards: dict[Cell, Board] = {
    (0, 0): sort_and_pad(squares_00),
    (0, 1): sort_and_pad(squares_01),
    (0, 2): sort_and_pad(squares_02),
    (1, 0): sort_and_pad(squares_10),
    (1, 1): sort_and_pad(squares_11),
    (1, 2): sort_and_pad(squares_12),
    (2, 0): sort_and_pad(squares_20),
    (2, 1): sort_and_pad(squares_21),
    (2, 2): sort_and_pad(squares_22),
}

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

letters = {
    (0, 0): {
        "top": [
            ("P", 15),
            ("A", 26),
            ("R", 43),
        ],
        "left": [
            ("T", 19),
            ("I", 34),
            ("L", 37),
        ],
    },
    (0, 1): {
        "top": [
            ("T", 0),
            ("R", 24),
            ("I", 41),
        ],
    },
    (0, 2): {
        "top": [
            ("D", 17),
            ("G", 20),
            ("E", 44),
        ],
    },
    (1, 0): {
        "left": [
            ("I", 15),
            ("N", 20),
            ("G", 39),
        ],
    },
}


# Solver, variables, objective, constraints
# ----------------------------------------------------------------------

STATUS_NAMES = {
    cp_model.UNKNOWN: "UNKNOWN",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.OPTIMAL: "OPTIMAL",
}


def solution(squares: Board) -> Solution | None:
    model = cp_model.CpModel()
    S = [[model.new_int_var(0, T, f"s{[k, l]}") for l in range(4)] for k in range(T)]

    X = [None for _ in range(T)]
    Y = [None for _ in range(T)]
    for k in range(T):
        side = square_sizes[k]
        x0, y0, x1, y1 = S[k]
        X[k] = model.new_interval_var(x0, side, x1, f"x{[k]}")
        Y[k] = model.new_interval_var(y0, side, y1, f"y{[k]}")

    model.add_no_overlap_2d(X, Y)

    for k, sq in enumerate(squares):
        if sq is None:
            continue
        x, y, _ = sq
        model.add(S[k][0] == x)
        model.add(S[k][1] == y)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    def get_square(k: int) -> Square:
        i0 = solver.value(S[k][0])
        j0 = solver.value(S[k][1])
        i1 = solver.value(S[k][2])
        return (i0, j0, i1 - i0)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return [get_square(k) for k in range(T)]
    else:
        return None


def solutions(boards: dict[Cell, Board]) -> dict[Cell, Solution | None]:
    return {(a, b): solution(squares) for (a, b), squares in boards.items()}


def plot(
    solutions: dict[Cell, Solution | None],
    letters: dict[Cell, dict[str, list[tuple[str, int]]]],
    filename: str = None,
) -> None:
    _, axes = plt.subplots(3, 3, figsize=(15, 15))
    for (a, b), solution_ in solutions.items():
        ax = axes[a][b]

        ax.set_xlim(0, T)
        ax.set_ylim(0, T)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Remove default ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot squares
        for square in solution_:
            i, j, s = square

            color = size_colors.get(s, "red")
            rect = patches.Rectangle((j, i), s, s, linewidth=2, edgecolor="black", facecolor=color, alpha=0.7)
            ax.add_patch(rect)

        # Draw red letters on top
        if a == 0:
            for pos in range(T):
                index = (pos + T * b) % 26
                letter = ALPHABET[index]
                ax.text(pos + 0.5, -2, letter, ha="center", va="center", fontsize=8, fontweight="bold", color="red")

        # Draw red letters on left
        if b == 0:
            for pos in range(T):
                index = (pos + T * a) % 26
                letter = ALPHABET[index]
                ax.text(
                    -2,
                    pos + 0.5,
                    letter,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="red",
                    rotation=90,
                )

        # Plot letters and add ticks
        grid_letters = letters.get((a, b), {})

        # Plot top letters and add x-ticks
        for letter, pos in grid_letters.get("top", []):
            ax.text(pos + 0.5, -2, letter, ha="center", va="center", fontsize=8, fontweight="bold")

        # Plot left letters and add y-ticks
        for letter, pos in grid_letters.get("left", []):
            ax.text(-2, pos + 0.5, letter, ha="center", va="center", fontsize=8, fontweight="bold", rotation=90)

        ax.invert_yaxis()

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


with Timer(initial_text="Solving all boards..."):
    sols = solutions(boards)
    plot(sols, letters)
# Solving all boards...
# Elapsed time: 12.4345 seconds

answer = "THE"


for (a, b), squares in sols.items():
    i, j, _ = squares[0]
    index_i = (i + T * a) % 26
    index_j = (j + T * b) % 26
    letter_i = ALPHABET[index_i]
    letter_j = ALPHABET[index_j]
    if (a, b) == (2, 0):
        answer += "A"
    answer += f"{letter_i}{letter_j}"

print(f"{answer = }")
# answer = 'THESUMOFCUBESISASQUARE'
