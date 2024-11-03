# Imports
# ----------------------------------------------------------------------
from itertools import product

import numpy as np
import pandas as pd

# States
# ----------------------------------------------------------------------
states = [
    ("California", 39538223),
    ("Texas", 29145505),
    ("Florida", 21538187),
    ("New York", 20201249),
    ("Pennsylvania", 13002700),
    ("Illinois", 12812508),
    ("Ohio", 11799448),
    ("Georgia", 10711908),
    ("North Carolina", 10439388),
    ("Michigan", 10077331),
    ("New Jersey", 9288994),
    ("Virginia", 8631393),
    ("Washington", 7705281),
    ("Arizona", 7151502),
    ("Massachusetts", 7029917),
    ("Tennessee", 6910840),
    ("Indiana", 6785528),
    ("Maryland", 6177224),
    ("Missouri", 6154913),
    ("Wisconsin", 5893718),
    ("Colorado", 5773714),
    ("Minnesota", 5706494),
    ("South Carolina", 5118425),
    ("Alabama", 5024279),
    ("Louisiana", 4657757),
    ("Kentucky", 4505836),
    ("Oregon", 4237256),
    ("Oklahoma", 3959353),
    ("Connecticut", 3605944),
    ("Utah", 3271616),
    ("Iowa", 3190369),
    ("Nevada", 3104614),
    ("Arkansas", 3011524),
    ("Mississippi", 2961279),
    ("Kansas", 2937880),
    ("New Mexico", 2117522),
    ("Nebraska", 1961504),
    ("Idaho", 1839106),
    ("West Virginia", 1793716),
    ("Hawaii", 1455271),
    ("New Hampshire", 1377529),
    ("Maine", 1362359),
    ("Rhode Island", 1097379),
    ("Montana", 1084225),
    ("Delaware", 989948),
    ("South Dakota", 886667),
    ("North Dakota", 779094),
    ("Alaska", 733391),
    # ('District of Columbia', 689545),
    ("Vermont", 643077),
    ("Wyoming", 576851),
]

states_df = pd.DataFrame(states, columns=["State", "Population"])
states_df["State"] = states_df["State"].str.replace(" ", "").str.upper()


# Grids
# ----------------------------------------------------------------------

example_grid = np.array(
    [
        ["T", "H", "O"],
        ["A", "I", "N"],
        ["E", "S", "L"],
    ],
    dtype=str,
)

grid = np.array(
    [
        ["T", "S", "H", "O", "L"],
        ["E", "C", "A", "L", "I"],
        ["S", "N", "I", "D", "N"],
        ["E", "R", "O", "F", "O"],
        ["W", "Y", "L", "S", "I"],
    ],
    dtype=str,
)


# Compute score
# ----------------------------------------------------------------------
def neigh(n: int, i: int, j: int) -> list[tuple[int, int]]:
    ans = []
    for di, dj in product(range(-1, 2), repeat=2):
        if di == dj == 0:
            continue
        if 0 <= i + di < n and 0 <= j + dj < n:
            ans.append((i + di, j + dj))
    return ans


def exists_state(grid: np.ndarray, state: str) -> bool:
    n = len(grid)
    num_chars = len(state)

    def dp(i: int, j: int, k: int, m: int) -> bool:
        if m < 0:
            return False
        if k == num_chars:
            return True
        for i_, j_ in neigh(n, i, j):
            replace = int(grid[i_, j_] != state[k])
            if dp(i_, j_, k + 1, m - replace):
                return True
        return False

    return any(dp(i, j, 0, 1) for i in range(n) for j in range(n))


def score(grid: np.ndarray, states: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    ans_df = states.copy()
    ans_df["exists"] = ans_df["State"].apply(lambda x: exists_state(grid, x))
    ans_score = ans_df["Population"][ans_df["exists"]].sum()
    return ans_score, ans_df


def submission(grid: np.ndarray) -> str:
    return "".join(grid.flatten()).lower()


# Solution
# ----------------------------------------------------------------------

min_points = 165379868


points, df = score(grid, states_df)
ans = submission(grid)

print(df, end="\n\n")
print(f"{points = }. It is {points > min_points} that points > min_points")
print(f"{ans = }")
