from math import factorial
from sympy import Float
from sympy.functions.combinatorial.numbers import stirling
from sympy.functions.combinatorial.factorials import binomial
from scipy.optimize import fsolve

# Events:
# W = I win
# C = I play continuous
# Ei = There are i robots other than me going discrete
# Fj = The robots other than me which go discrete occupy j races

# 1/3
# = P(W)
# = P(W|C)
# = Sum_{i = 0 to 23} P(Ei) P(W|C & Ei)
# = Sum_{i = 0 to 23} P(Ei) Sum_{j = 0 to 7} P(Fj|Ei) P(W|C & Ei & Fj)

def binom_pmf(i, n, p):
    return binomial(n, i) * p**i * (1 - p)**(n - i)

def p_e(i, p):
    return binom_pmf(i, 23, p)

def p_fe(i, j):
    return binomial(8, j) * stirling(i, j) * factorial(j) / 8**i

def p_wcef(i, j):
    return min(1, (8 - j)/(24 - i))

def p_w(p):
    return Float(sum(p_e(i, p) * sum(p_fe(i, j) * p_wcef(i, j) 
                     for j in range(0, 8)) for i in range(0, 24)))

p = fsolve(lambda p: p_w(p) - 1/3, 1)
print('p = {:.6f}'.format(p[0]))

# solution:
# p = 0.999560