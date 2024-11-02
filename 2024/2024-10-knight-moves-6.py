import numpy as np
from itertools import product
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# Types
# ----------------------------------------------------------------------
Square = tuple[int, int]
Path = list[Square]


# Input
# ----------------------------------------------------------------------

grid = np.array([
    [0, 1, 1, 2, 2, 2],
    [0, 1, 1, 2, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 0, 1, 1, 2],
    [0, 0, 0, 1, 1, 2],
])
n = grid.shape[0]

knight_moves = [
    ( 2,  1),
    ( 1,  2),
    (-1,  2),
    (-2,  1),
    (-2, -1),
    (-1, -2),
    ( 1, -2),
    ( 2, -1),
]

max_length = 15
max_sum = 10  # Possible to see that a + b + c < 10 can be found


# Paths
# ----------------------------------------------------------------------

def neighbours(square: Square) -> list[Square]:
    x, y = square
    squares = [(x + dx, y + dy) for dx, dy in knight_moves]
    inside = [(x, y) for x, y in squares if 0 <= x < n and 0 <= y < n]
    return inside


def path_continuations(path: Path) -> list[Path]:
    last_square = path[-1]
    next_squares = neighbours(last_square)
    return [path + [square] for square in next_squares if square not in path]


def paths(max_length: int) -> list[Path]:
    solutions = []
    starting_path = [(0, 0)]
    stack = [starting_path]
    while stack:
        path = stack.pop()
        if path[-1] == (n - 1, n - 1):
            solutions.append(path)
        elif len(path) < max_length:
            stack.extend(path_continuations(path))
    return solutions


def invert_path(path: Path) -> Path:
    return [(n - 1 - x, y) for x, y in path]


paths_blu = paths(max_length)
paths_red = [invert_path(path) for path in paths_blu]


# Iterator
# ----------------------------------------------------------------------

triples = list(product(range(1, max_sum), repeat=3))
ordered = list(sorted(triples, key=sum))
abcs = [(a, b, c) for a, b, c in ordered if a + b + c < max_sum]


# Score
# ----------------------------------------------------------------------

def score(triple: tuple[int, int, int], path: Path) -> int:
    values = [triple[grid[x, y]] for x, y in path]
    ans = values[0]
    for prev, curr in zip(values, values[1:]):
        ans = ans * curr if prev != curr else ans + curr
    return ans


def valid_score(triple: tuple[int, int, int], path: Path) -> bool:
    return score(triple, path) == 2024


for triple in tqdm(abcs):
    is_valid_score = partial(valid_score, triple)
    sol_blu = next(filter(is_valid_score, paths_blu), None)
    sol_red = next(filter(is_valid_score, paths_red), None)

    if sol_blu and sol_red:
        print(f'{triple = }')
        print(f'{sol_blu = }')
        print(f'{sol_red = }')
        break


def plot_sol(
    triple: tuple[int, int, int],
    path_blu: Path,
    path_red: Path,
) -> None:

    def plot_arrows(path: Path, color: str) -> None:
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
            plt.annotate(
                '',
                xy=(y2 + 0.5, x2 + 0.5),
                xytext=(y1 + 0.5, x1 + 0.5),
                arrowprops=dict(arrowstyle="->", color=color, lw=2)
            )

    annot = np.array([[triple[grid[x, y]] for y in range(n)] for x in range(n)])
    sns.heatmap(
        grid,
        annot=annot,
        cmap=None,
        yticklabels=['6', '5', '4', '3', '2', '1'],
        xticklabels=['a', 'b', 'c', 'd', 'e', 'f'],
    )
    plot_arrows(path_blu, 'blue')
    plot_arrows(path_red, 'red')

    plt.show()


def format_sol(
    triple: tuple[int, int, int],
    path_blu: Path,
    path_red: Path,
) -> str:
    xlabels = ['6', '5', '4', '3', '2', '1']
    ylabels = ['a', 'b', 'c', 'd', 'e', 'f']
    abc = list(map(str, triple))
    a1f6 = [f'{ylabels[y]}{xlabels[x]}' for x, y in path_red]
    a6f1 = [f'{ylabels[y]}{xlabels[x]}' for x, y in path_blu]
    soln = abc + a1f6 + a6f1
    return ','.join(soln)


print(format_sol(triple, sol_blu, sol_red))
plot_sol(triple, sol_blu, sol_red)
