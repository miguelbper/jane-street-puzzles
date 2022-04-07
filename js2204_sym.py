from sympy import *
from pprint import pprint
from string import ascii_lowercase


a, b, c, d, e, f, g, h, i, s = symbols('a, b, c, d, e, f, g, h, i, s')


m = Matrix([[a, b, c], 
            [d, e, f], 
            [g, h, i]])


eq = [
    s - sum(m.row(0)),
    s - sum(m.row(1)),
    s - sum(m.row(2)),
    s - sum(m.col(0)),
    s - sum(m.col(1)),
    s - sum(m.col(2)),
    s - trace(m),
    s - c + e + g
]


sol = next(iter(linsolve(eq, (a, b, c, e, f, h, s))))


ex = [
    sol[0], # a
    sol[1], # b
    sol[2], # c
    d,      # d
    sol[3], # e
    sol[4], # f
    g,      # g
    sol[5], # h
    i       # i
]


ineq = []

for ii in range(0,8):
    for jj in range(ii + 1, 8):
        ineq.append((ascii_lowercase[ii], ascii_lowercase[jj], ex[ii] - ex[jj]))