"""
Events:
W = I win
C = I play continuous
Ei = There are i robots other than me going discrete
Fj = The robots other than me which go discrete occupy j races

1/3
= P(W)
= P(W|C)
= Sum_{i = 0 to 23} P(Ei) P(W|C & Ei)
= Sum_{i = 0 to 23} P(Ei) Sum_{j = 0 to 7} P(Fj|Ei) P(W|C & Ei & Fj)
"""

from math import factorial

from scipy.optimize import fsolve
from sympy import Float
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import stirling


def binom_pmf(i, n, p):
    return binomial(n, i) * p**i * (1 - p) ** (n - i)


# prob_1(i, p) = P(Ei)
def prob_1(i, p):
    return binom_pmf(i, 23, p)


# prob_2(i, j) = P(Fj|Ei)
def prob_2(i, j):
    return binomial(8, j) * stirling(i, j) * factorial(j) / 8**i


# prob_3(i, j) = P(W|C & Ei & Fj)
def prob_3(i, j):
    return min(1, (8 - j) / (24 - i))


# prob_w(p) = P(W)
def prob_w(p):
    return Float(
        sum(
            prob_1(i, p) * sum(prob_2(i, j) * prob_3(i, j) for j in range(0, 8))
            for i in range(0, 24)
        )
    )


# solve for p
p = fsolve(lambda p: prob_w(p) - 1 / 3, 1)
print(f"p = {p[0]:.6f}")

# solution:
# p = 0.999560
