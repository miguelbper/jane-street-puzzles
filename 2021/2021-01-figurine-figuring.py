"""There is 1 ball of color 1 2 balls of color 2 ... 12 balls of color 12 The
total number of balls is 1 + 2 + ... + 12 = 78.

Throughout, we will have the following indices
    i = maximum number of balls of the same color drawn before ball 1
    j = position of ball 1
    k = an index saying which bag balls were removed from
    l = number of balls drawn from bag k

Define random variables
    N = maximum number of balls of the same color drawn before ball 1
    X = number of balls drawn before ball 1

We compute the expected value using the definition:
    E
    = Σ_{i=0,...,12} i P(N=i)
    = Σ_{i=0,...,12} i Σ_{j=0,...,77} P(N=i | X=j) P(X=j)
    = (1/78) * Σ_{i=0,...,12} i Σ_{j=0,...,77} P(N=i | X=j),
where
    P(N=i | X=j)
    = # (arranjements of j balls s.t. max balls of same color = i) /
      # (arranjements of j balls)

To count, we will use generating functions. Define
    f_{i,k}(x) = Σ_{l=0,...,min(i,k)} binom(k, l) x**l
    g_i(x)     = Π_{k=2,...,12} f_{i,k}(x) = Σ_{n=0,...,Ni} c_{i,n} x**n
where
    Ni      = deg g_i
            = maximum possible number of balls that could be drawn
              so that the max number of balls of same color is i
    c_{i,n} = nth coefficient of gi
            = # (arranj. of n balls s.t. max balls of same color <= i)

Therefore,
    P(N=i | X=j) = (c_{i,j} - c_{i-1,j}) / c_{12,j}
"""

import numpy as np
import sympy as sp
from sympy.abc import x


# define polynomial f_{i,k}
def f(i, k):
    fs = [sp.binomial(k, l) for l in range(min(i, k) + 1)]
    return sp.Poly.from_list(fs[::-1], x)


# define polynomial g_{i}
def g(i):
    return sp.prod(f(i, k) for k in range(2, 13))


# compute C(i) = list of coefficients of polynomial g(i)
def C(i):
    poly = g(i)
    coef = poly.all_coeffs()[::-1]
    return coef + [0] * (77 - poly.degree(x))


# compute the expected value, using the formulas explained above
CC = np.array([C(i) for i in range(13)])
C0 = CC[0:12]
C1 = CC[1:13]
P = (C1 - C0) / CC[12]
e = np.sum(np.array(range(1, 13)) @ P) / 78
print(f"e = {e:.6f}")
# e = 6.859787
