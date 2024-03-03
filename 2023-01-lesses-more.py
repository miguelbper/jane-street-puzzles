from sympy.ntheory import primefactors
from functools import cache
import numpy as np
from math import gcd


# Functions
# ----------------------------------------------------------------------

def r(a, b, c, d):
    k = min(a, b, c, d)
    a, b, c, d = a - k, b - k, c - k, d - k
    k = gcd(a, b, c, d)
    a, b, c, d = a // k, b // k, c // k, d // k
    return a, b, c, d


def n(a, b, c, d):
    return (abs(b - a), abs(c - b), abs(d - c), abs(a - d))


@cache
def f(a, b, c, d):
    if (a, b, c, d) == (0, 0, 0, 0):
        return 1
    return 1 + f(*n(a, b, c, d))


# Experiment: solve problem for different (small) values of N = max(x)
# ----------------------------------------------------------------------
print('Experiment')
N = 50
factors = [set(primefactors(x)) for x in range(N + 1)]
factors[0] = set.union(*factors[1:])

while N:

    M = 1
    S = 4 * N
    x = (0, 0, 0, 0)

    for b in range(N + 1):
        for c in range(N + 1):
            factors_bc = factors[b].intersection(factors[c])
            for d in range(b, N + 1):
                if set() == factors[d].intersection(factors_bc):
                    m = f(0, b, c, d)
                    if m > M or (m == M and 0 + b + c + d < S):
                        x = (0, b, c, d)
                        M = m
                        S = sum(x)

    a, b, c, d = x
    p, q, u, v = r(*n(*x))
    print(f'N = {N:2d}, max(x) = {max(x):2d}, M = {M:2d}, '
          + f'x = ({a}, {b:d}, {c:2d}, {d:2d}), '
          + f'r.n(x) = ({p}, {q:d}, {u:2d}, {v:2d})')
    N = max(0, max(x) - 1)

'''
Experiment
N = 50, max(x) = 44, M = 14, x = (0, 7, 20, 44), r.n(x) = (0, 6, 17, 37)
N = 43, max(x) = 37, M = 13, x = (0, 6, 17, 37), r.n(x) = (0, 5, 14, 31)
N = 36, max(x) = 31, M = 12, x = (0, 5, 14, 31), r.n(x) = (0, 2,  6, 13)
N = 30, max(x) = 13, M = 11, x = (0, 2,  6, 13), r.n(x) = (0, 2,  5, 11)
N = 12, max(x) = 11, M = 10, x = (0, 2,  5, 11), r.n(x) = (0, 1,  4,  9)
N = 10, max(x) =  9, M =  9, x = (0, 1,  4,  9), r.n(x) = (0, 1,  2,  4)
N =  8, max(x) =  4, M =  8, x = (0, 1,  2,  4), r.n(x) = (0, 0,  1,  3)
N =  3, max(x) =  3, M =  7, x = (0, 0,  1,  3), r.n(x) = (0, 1,  2,  3)
N =  2, max(x) =  1, M =  5, x = (0, 0,  0,  1), r.n(x) = (0, 0,  1,  1)

Let x_M be the value of x that corresponds to each M.
Notice that:
    1. x_M is nondecreasing
    2. r(n(x_{M+1})) = x_M, for M >= 7
'''


# Solution
# ----------------------------------------------------------------------
'''
We solve the problem assuming that the properties 1 and 2 above are true
in general.

Let
    x_{M+1} = (a, b, c, d) = (0, b, c, d)
    x_M     = (p, q, u, v) = (0, q, u, v)

Then,
    (0, q, u, v)
        = x_M
        = r(n(x_{M+1}))
        = r(n(0, b, c, d))
        = r(b, c - b, d - c, d)
        = (0, c - 2*b, d - c - b, d - b) / k,
    where k = gcd(0, c - 2*b, d - c - b, d - b)

    =>

    |b|    k  | -1 -1 1 | |q|
    |c| = ___ |  0 -2 2 | |u|
    |d|    2  | -1 -1 3 | |v|
    (these are supposed to be matrices/column vectors)

We can choose k = 1 + any(n % 2 for n in [b, c, d]).
This equation allows us to find x_{M+1} from x_M. Keep computing x_{M+1}
while max(x_{M+1}) <= 10**7.
'''

print('\nSolution')

N = 10**7
x = ans = (0, 1, 0, 1)
M = f(*x)
A = np.array([
    [-1, -1, 1],
    [ 0, -2, 2],
    [-1, -1, 3],
])

while max(x) <= N:
    ans = x
    p, q, u, v = x
    print(f'M = {M:2d}    =>    x = ({p}, {q:7d}, {u:7d}, {v:7d})')

    b, c, d = A @ np.array((q, u, v))
    k = 1 + any(n % 2 for n in [b, c, d])
    b, c, d = (np.array([b, c, d]) * k) // 2

    x = (0, b, c, d)
    M += 1

print(f'\nans = {ans[0]};{ans[1]};{ans[2]};{ans[3]}'
      + f'    =>    M = f(ans) = {f(*ans)}')
'''
Solution
M =  3    =>    x = (0,       1,       0,       1)
M =  4    =>    x = (0,       0,       1,       1)
M =  5    =>    x = (0,       0,       0,       1)
M =  6    =>    x = (0,       1,       2,       3)
M =  7    =>    x = (0,       0,       1,       3)
M =  8    =>    x = (0,       1,       2,       4)
M =  9    =>    x = (0,       1,       4,       9)
M = 10    =>    x = (0,       2,       5,      11)
M = 11    =>    x = (0,       2,       6,      13)
M = 12    =>    x = (0,       5,      14,      31)
M = 13    =>    x = (0,       6,      17,      37)
M = 14    =>    x = (0,       7,      20,      44)
M = 15    =>    x = (0,      17,      48,     105)
M = 16    =>    x = (0,      20,      57,     125)
M = 17    =>    x = (0,      24,      68,     149)
M = 18    =>    x = (0,      57,     162,     355)
M = 19    =>    x = (0,      68,     193,     423)
M = 20    =>    x = (0,      81,     230,     504)
M = 21    =>    x = (0,     193,     548,    1201)
M = 22    =>    x = (0,     230,     653,    1431)
M = 23    =>    x = (0,     274,     778,    1705)
M = 24    =>    x = (0,     653,    1854,    4063)
M = 25    =>    x = (0,     778,    2209,    4841)
M = 26    =>    x = (0,     927,    2632,    5768)
M = 27    =>    x = (0,    2209,    6272,   13745)
M = 28    =>    x = (0,    2632,    7473,   16377)
M = 29    =>    x = (0,    3136,    8904,   19513)
M = 30    =>    x = (0,    7473,   21218,   46499)
M = 31    =>    x = (0,    8904,   25281,   55403)
M = 32    =>    x = (0,   10609,   30122,   66012)
M = 33    =>    x = (0,   25281,   71780,  157305)
M = 34    =>    x = (0,   30122,   85525,  187427)
M = 35    =>    x = (0,   35890,  101902,  223317)
M = 36    =>    x = (0,   85525,  242830,  532159)
M = 37    =>    x = (0,  101902,  289329,  634061)
M = 38    =>    x = (0,  121415,  344732,  755476)
M = 39    =>    x = (0,  289329,  821488, 1800281)
M = 40    =>    x = (0,  344732,  978793, 2145013)
M = 41    =>    x = (0,  410744, 1166220, 2555757)
M = 42    =>    x = (0,  978793, 2779074, 6090307)
M = 43    =>    x = (0, 1166220, 3311233, 7256527)
M = 44    =>    x = (0, 1389537, 3945294, 8646064)

ans = 0;1389537;3945294;8646064    =>    M = f(ans) = 44
'''
