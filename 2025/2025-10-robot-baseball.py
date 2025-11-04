import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sympy import diff, solve, symbols
from sympy.abc import a, b, p

# Symbolic calculations
# ------------------------------------------------------------------------------
"""
Notation:

A = batter
B = pitcher

a = strategy of player A = probability of wait
b = strategy of player B = probability of ball

i = number of balls
j = number of strikes

x = x[i, j] = expected score in state (i, j)
q = q[i, j] = probability of full count in state (i, j)

xi = x[i + 1, j]
xj = x[i, j + 1]
qi = q[i + 1, j]
qj = q[i, j + 1]
"""


xi = symbols("xi")
xj = symbols("xj")
qi = symbols("qi")
qj = symbols("qj")


aa = np.array([a, 1 - a])
bb = np.array([b, 1 - b])
ab = np.outer(aa, bb)

# fmt: off
xm = np.array([
    [xi, xj],
    [xj, (1 - p) * xj + 4 * p],
])

qm = np.array([
    [qi, qj],
    [qj, (1 - p) * qj],
])
# fmt: on

x = np.sum(ab * xm)  # Law of total expectation
q = np.sum(ab * qm)  # Law of total probability
print(f"x = {x}")
print(f"q = {q}")

# Find Nash equilibrium: partial derivatives of x w.r.t. a and b should be 0
dx_da = diff(x, a)
b0 = solve(dx_da, b)[0]
print(f"b0 = {b0}")

dx_db = diff(x, b)
a0 = solve(dx_db, a)[0]
print(f"a0 = {a0}")

# x0 = x(a0, b0), value of x at Nash equilibrium
x0 = x.subs({a: a0, b: b0}).simplify()
print(f"x0 = {x0}")

# Dynamic programming
# ------------------------------------------------------------------------------

# Arrays to store values for each state (i, j)
X = np.zeros((5, 4), dtype=np.object_)
A = np.zeros((5, 4), dtype=np.object_)
B = np.zeros((5, 4), dtype=np.object_)
Q = np.zeros((5, 4), dtype=np.object_)

X[4] = 1

for i in reversed(range(4)):
    for j in reversed(range(3)):
        print(f"Computing expressions for (i, j) = ({i}, {j})")
        substitutions = {xi: X[i + 1, j], xj: X[i, j + 1]}
        X[i, j] = x0.subs(substitutions).simplify()
        A[i, j] = a0.subs(substitutions).simplify()
        B[i, j] = b0.subs(substitutions).simplify()

        if (i, j) == (3, 2):
            Q[i, j] = 1
        else:
            Q[i, j] = q.subs({qi: Q[i + 1, j], qj: Q[i, j + 1], a: A[i, j], b: B[i, j]}).simplify()


# Optimization (find maximal q(p))
# ------------------------------------------------------------------------------

f = -Q[0, 0]
df = diff(f, p)


def fun(x: NDArray) -> float:
    x_ = x.item()
    f_x = f.subs(p, x_)
    return float(f_x)


def jac(x: NDArray) -> NDArray:
    x_ = x.item()
    df_x = df.subs(p, x_)
    return np.array([float(df_x)])


optimization = minimize(fun, [0.5], jac=jac, tol=1e-20)
p0 = optimization.x.item()
q0 = -optimization.fun
print(f"{p0 = :.10f}")
print(f"{q0 = :.10f}")


# Plot
# ------------------------------------------------------------------------------

ps = np.arange(0, 1, 0.01)
qs = np.array([Q[0, 0].subs({p: p_}) for p_ in ps])

plt.plot(ps, qs)
plt.scatter(p0, q0, color="red")
plt.show()
"""
x = a*b*xi + a*xj*(1 - b) + b*xj*(1 - a) + (1 - a)*(1 - b)*(4*p + xj*(1 - p))
q = a*b*qi + a*qj*(1 - b) + b*qj*(1 - a) + qj*(1 - a)*(1 - b)*(1 - p)
b0 = p*(xj - 4)/(p*xj - 4*p - xi + xj)
a0 = p*(xj - 4)/(p*xj - 4*p - xi + xj)
x0 = (p*xi*xj - 4*p*xi - xi*xj + xj**2)/(p*xj - 4*p - xi + xj)

p0 = 0.2269732325
q0 = 0.2959679934 (answer)
"""
