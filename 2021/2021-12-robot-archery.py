"""We may assume that the radius of the target is 1 and that the distance of
the arrow to the center has a uniform distribution.

Let p_{n,k}(x) be the probability that robot k is the winner, in a game
with n robots, given that x is the best current throw.

We wish to compute p_{4,4}(1).

We will write a system of ODEs for p_{n,k}(x) and solve that system
symbolically (with sympy). We will assume n >= 2.

One of the robots must win:
    1 = p_{n,1}(x) + ... + p_{n,n}(x)

Define the following events:
    W(n,k,x) = robot k is the winner, in a game with n robots,
               given that x is the best current throw
    T(u)     = robot 1 throws u

Integral equation 1:
    p_{n,1}(x)
    = P(W(n,1,x))
    = Σ_u P(W(n,1,x) | T(u))       [*]
    = Σ_{u < x} P(W(n,1,x) | T(u))
    = Σ_{u < x} P(W(n,n,u))
    = ∫_{0<u<x} p_{n,n}(u) du
* - this should actually be a continuous sum, I'm being imprecise here
    this could be made rigorous with a limiting argument

Integral equation 2: If 1 < k < n - 1,
    p_{n,k}(x)
    = P(W(n,k,x))
    = Σ_{u<x} P(W(n,k,x) | T(u)) + Σ_{u>x} P(W(n,k,x) | T(u))
    = Σ_{u<x} P(W(n,k-1,u)) + Σ_{u>x} P(W(n-1,k-1,x))
    = ∫_{0<u<x} p_{n,k-1}(u) du + (1-x) p_{n-1,k-1}(x)

We now write the ODE system that we must solve.

Initial conditions: for all 0 < k < n-1, (because of the two eqs with ∫)
    p_{n,k}(0) = 0

ODEs: differentiating the two eqs with ∫,
    p'_{n,1} = p_{n,n} = 1 - p_{n,1} - ... - p_{n,n}       [k = 1]
    p'_{n,k} = p_{n,k-1} - p_{n-1,k-1} + (1-x) p_{n-1,k-1} [1 < k < n-1]
"""

from sympy import Eq, Function, diff, dsolve, simplify, symbols
from sympy.solvers.ode.systems import dsolve_system

# define symbols
# ----------------------------------------------------------------------

p21 = symbols("p21", cls=Function)
p31, p32 = symbols("p31 p32", cls=Function)
p41, p42, p43 = symbols("p41 p42 p43", cls=Function)
x = symbols("x")


# solve equations for n = 2
# ----------------------------------------------------------------------

e21 = Eq(p21(x).diff(x), 1 - p21(x))  # write ODE
sol = dsolve(e21, ics={p21(0): 0})  # solve ODE
p21 = sol.rhs  # write solution as expression
print(f"p21(x) = {p21}\n")  # print solution
# p21(x) = 1 - exp(-x)

# solve equations for n = 3 (system)
# ----------------------------------------------------------------------

# write ODE system
e31 = Eq(p31(x).diff(x), 1 - p31(x) - p32(x))
e32 = Eq(p32(x).diff(x), p31(x) - p21 + (1 - x) * diff(p21, x))
# solve ODE system
sol = dsolve_system([e31, e32], ics={p31(0): 0, p32(0): 0}, doit=True)
# write solutions as expressions and simplify
p31 = simplify(sol[0][0].rhs)
p32 = simplify(sol[0][1].rhs)
# print solutions
print(f"p31(x) = {p31}")
print(f"p32(x) = {p32}\n")
# p31(x) = x*exp(-x) + 1 - exp(-x) - 2*sqrt(3)*exp(-x/2)*sin(sqrt(3)*x/2)/3
# p32(x) = -exp(-x) + 2*sqrt(3)*exp(-x/2)*sin(sqrt(3)*x/2 + pi/3)/3

# solve equations for n = 4
# ----------------------------------------------------------------------

# write ODE system
e41 = Eq(p41(x).diff(x), 1 - p41(x) - p42(x) - p43(x))
e42 = Eq(p42(x).diff(x), p41(x) - p31 + (1 - x) * p31.diff(x))
e43 = Eq(p43(x).diff(x), p42(x) - p32 + (1 - x) * p32.diff(x))
eqs = [e41, e42, e43]
ics = {p41(0): 0, p42(0): 0, p43(0): 0}
# solve ODE system
sol = dsolve_system(eqs, ics=ics, doit=True)
# write solutions as expressions and simplify
p41 = simplify(sol[0][0].rhs)
p42 = simplify(sol[0][1].rhs)
p43 = simplify(sol[0][2].rhs)
p44 = simplify(1 - p41 - p42 - p43)
# print solutions
print(f"p41(x) = {p41}")
print(f"p42(x) = {p42}")
print(f"p43(x) = {p43}")
print(f"p44(x) = {p44}\n")
# p41(x) = -3*x**2*exp(-x)/4 + 2*x*exp(-x) - 2*x*exp(-x/2)*cos(sqrt(3)*x/2
#          + pi/3) + 5*sqrt(2)*cos(x + pi/4)/4 + 1 - 5*exp(-x)/4
#          - 2*sqrt(3)*exp(-x/2)*sin(sqrt(3)*x/2 + pi/3)/3

# p42(x) = (-8*(sqrt(3)*x*sin(sqrt(3)*x/2 + pi/3) + 3*sin(sqrt(3)*x/2
#          + pi/6))*exp(x) + 3*(-x**2 + 6*x - 1)*exp(x/2)
#          + 15*sqrt(2)*exp(3*x/2)*sin(x + pi/4))*exp(-3*x/2)/12

# p43(x) = (3*(x**2 - 7)*exp(x/2) + 4*(-2*sqrt(3)*x*sin(sqrt(3)*x/2)
#          - sqrt(3)*sin(sqrt(3)*x/2) + 9*cos(sqrt(3)*x/2))*exp(x)
#          - 15*sqrt(2)*exp(3*x/2)*cos(x + pi/4))*exp(-3*x/2)/12

# p44(x) = 3*x**2*exp(-x)/4 + 2*x*exp(-x/2)*cos(sqrt(3)*x/2)
#          - 7*x*exp(-x)/2 - 5*sqrt(2)*sin(x + pi/4)/4 + 13*exp(-x)/4
#          + 5*sqrt(3)*exp(-x/2)*sin(sqrt(3)*x/2)/3
#          - exp(-x/2)*cos(sqrt(3)*x/2)

# compute p44(1) = Darrin's probability of winning
p_exact = p44.subs({x: 1})
p_decimal = p_exact.evalf(15)
print(f"p44(1) = {p_exact}\n       = {p_decimal:.11f}")
# p44(1) = -5*sqrt(2)*sin(pi/4 + 1)/4 + exp(-1/2)*cos(sqrt(3)/2)
#          + exp(-1)/2 + 5*sqrt(3)*exp(-1/2)*sin(sqrt(3)/2)/3
#        = 0.18343765086
