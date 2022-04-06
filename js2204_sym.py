from sympy import *
from pprint import pprint

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

sol = linsolve(eq, (a, b, c, e, f, h, s))