"""Let:

    - W(x) = "R1 wins, given that the marker starts at x" (event)
    - D(u) = "The first draw of R1 is u"                  (event)
    - p(x) = P(W(x))                                      (function of x)

We will deduce an ODE for p(x), solve that ODE with sympy, and finally
solve 1/2 = p(x).

Integral equation:
    p(x)
    = P(W(x))
    = Σ_u P(W(x) | D(u)) P(D(u))        (*)
    = Σ_{x+u>1/2} P(W(x)|D(u))P(D(u)) + Σ_{x+u<1/2} P(W(x)|D(u))P(D(u))
    = Σ_{x+u>1/2} P(D(u)) + Σ_{x+u<1/2} (1 - P(W(-x-u)))P(D(u))
    = 1 - Σ_{x+u<1/2} P(W(-x-u)) P(D(u))
    = 1 - ∫_{0 < u < 1/2-x} p(-x-u) du
    = 1 - ∫_{x < v < 1/2} p(-v) dv

* - this should actually be a continuous sum
    the computation could be made rigorous with a limiting argument

ODE (take derivative of integal equation):
    p'(x) = p(-x)
    => p''(x) = - p'(-x) = - p(x)

Boundary conditions:
    p(1/2) = 1              (by the integral equation)
    p'(-1/2) = p(1/2) = 1   (by the ODE + 1st boundary condition)
"""

from fractions import Fraction
from math import asin, pi, sin

from sympy import Eq, Function, dsolve, simplify, symbols

# define symbols
p = symbols("p", cls=Function)
x = symbols("x")

# define ODE to solve
eq = Eq(p(x).diff(x, x), -p(x))

# define initial conditions
ics = {
    p(Fraction(1, 2)): 1,
    p(x).diff(x).subs(x, -Fraction(1, 2)): 1,
}

# solve ODE
sol = dsolve(eq, ics=ics)
p = simplify(sol.rhs)
print(f"p(x) = {p}")
# P(x) = sin(x + pi/4)/sin(1/2 + pi/4)

# Therefore,
# p(x) = 1/2
# => sin(x + pi/4)/sin(1/2 + pi/4) = 1/2
# => sin(x + pi/4) = sin(1/2 + pi/4)/2
# => x + pi/4 = arcsin(sin(1/2 + pi/4)/2)
# => x = arcsin(sin(1/2 + pi/4)/2) - pi/4
#      = -0.2850001

# solve for x
x = asin(sin(1 / 2 + pi / 4) / 2) - pi / 4
print(f"x = {x:.7f}")
