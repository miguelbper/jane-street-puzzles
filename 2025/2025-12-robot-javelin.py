import numpy as np
from sympy import Expr, Rational, diff, simplify, solve
from sympy.abc import d, x, y

"""
Notation:
    A = Java-lin
    B = Spears Robot
    Xn = nth throw of A
    Yn = nth throw of B
    x = A's rethrow threshold (A rethrows iff X1 < x)
    y = B's rethrow threshold (B rethrows iff Y1 < y)
    d = threshold B uses to spy on A's first throw
    p(x, y) = P(A wins)
"""


def get_p_game_1() -> Expr:
    """
    Game 1: Fair game (Nash equilibrium).

    Both players use threshold strategies. Computes P(A wins) as a function of
    thresholds x and y using law of total probability over 4 cases:
        (X1 < x or X1 >= x) x (Y1 < y or Y1 >= y)
    """
    U = np.array([x * y, x * (1 - y), (1 - x) * y, (1 - x) * (1 - y)])  # P(case)
    V = np.array([1, 1 - y, 1 + x, (1 - y) / (1 - x)]) * Rational(1, 2)  # P(A_x,y | case)
    return simplify(np.dot(U, V))


def get_x_nash_equilibrium() -> Expr:
    """
    Compute Nash equilibrium threshold x* = (sqrt(5) - 1) / 2 (golden ratio).

    By symmetry, both players use the same threshold at equilibrium.
    Find x* = argmax_x min_y p(x,y) by setting dp/dx = 0 with y = x.
    """
    p = get_p_game_1()
    dpdx = diff(p, x)
    dpdx_subs = dpdx.subs({y: x})
    x0 = solve(dpdx_subs, x)[0]
    return x0


def get_p_game_2() -> Expr:
    """
    Game 2: B exploits information leak (A plays Nash, B adapts).

    B learns whether A's first throw is below d (= Nash threshold).
    - If X1 < d: B knows A will rethrow, so B uses threshold 0.5.
    - If X1 >= d: B knows A keeps X1, so B uses threshold y.

    Computes P(A wins) with A at Nash equilibrium, as function of y.
    """
    U = np.array([x / 2, x / 2, (1 - x) * y, (1 - x) * (1 - y)])  # P(case)
    V = np.array([1, Rational(1, 2), 1 + x, (1 - y) / (1 - x)]) * Rational(1, 2)  # P(A_x,y | case)
    return simplify(np.dot(U, V))


def get_y_optimal_for_b() -> Expr:
    """
    Compute B's optimal threshold y* = (5 - sqrt(5)) / 4 when X1 >= d.

    B minimizes P(A wins) by choosing y to minimize p(x0, y).
    """
    x0 = get_x_nash_equilibrium()
    p = get_p_game_2().subs({x: x0})
    dpdy = diff(p, y)
    y0 = solve(dpdy, y)[0]
    return y0


def get_p_game_3() -> Expr:
    """
    Game 3: A counter-exploits B's strategy (A adapts, B thinks A plays Nash).

    B uses threshold 0.5 when X1 < d, and threshold y when X1 >= d.
    A deviates from Nash by using threshold x < d to exploit B's strategy.

    6 cases (assuming x < d):
        1. X1 < x, Y1 < 0.5:    A rethrows, B rethrows
        2. X1 < x, Y1 >= 0.5:   A rethrows, B keeps Y1
        3. x <= X1 < d, Y1 < 0.5:  A keeps X1, B rethrows
        4. x <= X1 < d, Y1 >= 0.5: A keeps X1, B keeps Y1
        5. X1 >= d, Y1 < y:     A keeps X1, B rethrows
        6. X1 >= d, Y1 >= y:    A keeps X1, B keeps Y1
    """
    U = np.array(
        [
            x / 2,
            x / 2,
            (d - x) / 2,
            (d - x) / 2,
            (1 - d) * y,
            (1 - d) * (1 - y),
        ]
    )  # P(case)
    V = np.array(
        [
            Rational(1, 2),
            Rational(1, 4),
            (d + x) / 2,
            d + x - 1,
            (1 + d) / 2,
            (1 - y) / (2 * (1 - d)),
        ]
    )  # P(A_x,y | case)
    return simplify(np.dot(U, V))


def get_p_optimal_for_a() -> Expr:
    """Compute A's optimal win probability when counter-exploiting B.

    Substitutes d = Nash threshold and y = B's optimal threshold into Game 3,
    then finds x* = 7/12 that maximizes P(A wins).

    Returns P(A wins) = (229 - 60*sqrt(5)) / 192 ≈ 0.4939.
    """
    x0 = get_x_nash_equilibrium()
    y0 = get_y_optimal_for_b()
    p = get_p_game_3().subs({d: x0, y: y0})
    dpdx = diff(p, x)
    x0 = solve(dpdx, x)[0]
    p0 = p.subs({x: x0}).simplify()
    return p0


x0 = get_x_nash_equilibrium()
y0 = get_y_optimal_for_b()
p0 = get_p_optimal_for_a()

width = max(len(str(var)) for var in [x0, y0, p0])
print(f"x0 = {str(x0).ljust(width)} = {x0.evalf()}")
print(f"y0 = {str(y0).ljust(width)} = {y0.evalf()}")
print(f"p0 = {str(p0).ljust(width)} = {p0.evalf()}")
