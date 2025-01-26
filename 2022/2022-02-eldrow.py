import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from codetiming import Timer
from numpy.typing import NDArray
from z3 import And, BoolRef, Implies, IntVector, Not, Or, Tactic, sat, set_param

set_param("parallel.enable", True)


def int_from_char(c: str) -> int:
    return ord(c) - ord("A")


def char_from_int(i: int) -> str:
    return chr(i + ord("A"))


words = pd.read_csv(Path(__file__).parent.parent / "utils" / "eldrow_wordlist.csv", names=["col"])["col"]
digits = np.array([list(map(int_from_char, word)) for word in words])


def eldrow(n: int, matches: list[str]) -> tuple[NDArray[np.int32], NDArray[np.bool_], NDArray[np.bool_]] | None:
    s = Tactic("qffd").solver()
    X = np.array(IntVector("x", n * 5)).reshape((n, 5))

    matches = np.array([[int_from_char(c) for c in word] for word in matches])
    num_matches = len(matches)
    target = X[-1]

    s += [And(x >= 0, x < 26) for x in X.flat]
    s += [Not(And([w == t for w, t in zip(word, target)])) for word in X[:-1]]
    s += [Or([And([r == d for r, d in zip(word, match)]) for match in digits]) for word in X[: n - num_matches]]
    s += [And([w == m for w, m in zip(word, match)]) for word, match in zip(X[-num_matches:], matches)]

    def is_yellow(i: int, j: int) -> BoolRef:
        char = X[i, j]
        word = X[i]
        num_target = sum(t == char for t in target)
        num_word = sum(And(w == char, True if k < j else t == char) for k, (w, t) in enumerate(zip(word, target)))
        exists = Or([t == char for k, t in enumerate(target) if k != j])
        green = target[j] == char
        return And(Not(green), exists, num_target > num_word)

    G = np.array([[X[i, j] == X[n - 1, j] for j in range(5)] for i in range(n)])
    Y = np.array([[is_yellow(i, j) for j in range(5)] for i in range(n)])
    Z = np.array([[And(Not(G[i, j]), Not(Y[i, j])) for j in range(5)] for i in range(n)])

    for i0 in range(n - 1):
        for i1 in range(i0 + 1, n):
            word, middle = X[i0], X[i1]
            for j in range(5):
                char = word[j]
                green = G[i0, j]
                exists_gray_char = Or([And(gray, w == char) for w, gray in zip(word, Z[i0])])
                num_word = sum(And(w == char, Not(gray)) for w, gray in zip(word, Z[i0]))
                num_middle = sum(m == char for m in middle)
                s += green == (middle[j] == char)
                s += Implies(Not(green), num_middle >= num_word)
                s += Implies(And(Not(green), exists_gray_char), num_middle == num_word)

    if s.check() == sat:
        m = s.model()
        xm = np.vectorize(lambda x: m.evaluate(x).as_long())(X)
        gm = np.vectorize(lambda x: bool(m.evaluate(x)))(G)
        ym = np.vectorize(lambda x: bool(m.evaluate(x)))(Y)
        return xm, gm, ym
    return None


def plot(cm: NDArray[np.str_], gm: NDArray[np.bool_], ym: NDArray[np.bool_]) -> None:
    plt.figure(figsize=(8, n))

    data = np.zeros((n, 5))
    data[ym] = 1
    data[gm] = 2
    colors = ["gray", "yellow", "green"]

    ax = sns.heatmap(data, annot=cm, fmt="", cmap=colors, cbar=False, square=True, vmin=0, vmax=2)

    for text in ax.texts:
        val = float(text.get_position()[0])
        text.set_color("white" if data[int(text.get_position()[1]), int(val)] == 0 else "black")

    plt.title(f"Eldrow Solution (n={n})")
    plt.show()


# Search for solutions whose last words match a given pattern
fixed = ".OWER"
matches = [word for word in words if re.match(f"^{fixed}$", word)]
print(f"Found matches: {matches}")

n = 16
with Timer(initial_text=f"Finding eldrow with length = {n}..."):
    soln = eldrow(n, matches)
if soln is not None:
    xm, gm, ym = soln
    cm = np.vectorize(lambda x: char_from_int(x))(xm)
    solution = ["".join(row) for row in cm]
    print(f"{n = :2d} => {solution = }")
    plot(cm, gm, ym)
# Found matches: ['COWER', 'LOWER', 'MOWER', 'POWER', 'ROWER', 'SOWER', 'TOWER']
# Finding eldrow with length = 16...
# Solving...
# Elapsed time: 20.2936 seconds
# n = 16 => solution = ['QUEUE', 'GAZER', 'INNER', 'ODDER', 'HOVER', 'JOKER', 'FOYER', 'BOXER',
#                       'WOOER', 'COWER', 'LOWER', 'MOWER', 'POWER', 'ROWER', 'SOWER', 'TOWER']
