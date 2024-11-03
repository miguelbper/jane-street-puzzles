"""Stirling(n, k) = number of partitions of {1,...,n} into k sets.

prob(robot playing uniform wins, everyone else plays discrete)
= 1 - prob(robot playing uniform loses, everyone else plays discrete)
= 1 - prob(every race is filled)
= 1 - # (ways of placing 3n-1 robots into n races, filling every race)
      / # (ways of placing 3n-1 robots into n races)
"""

from math import factorial

from sympy import Float
from sympy.functions.combinatorial.numbers import stirling


def p(n):
    return Float(1 - factorial(n) * stirling(3 * n - 1, n) / n ** (3 * n - 1))


# print values
for n in range(1, 100):
    print(f"n = {n}, p = {p(n):.6f}")
    if p(n) > 1 / 3:
        break

# solution:
# n = 1, p = 0.000000
# n = 2, p = 0.062500
# n = 3, p = 0.116598
# n = 4, p = 0.166012
# n = 5, p = 0.212093
# n = 6, p = 0.255368
# n = 7, p = 0.296128
# n = 8, p = 0.334578
