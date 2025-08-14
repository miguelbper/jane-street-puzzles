"""
Consider the probability space of pairs (u, v), where u, v in [1, 2] and u < v are the speeds of the cars, with a
probability density function p(u, v) that we will compute later.

Define the random variable
    D_a := "Distance that u loses due to being passed by v in a highway with parameter a"
         = { u^2        if u < a   (slow lane)
           { (u - a)^2  otherwise  (fast lane)

We wish to compute
    f(a) := E[D_a]
          = ∫∫_{1 <= u < v <= 2} D_a(u, v) p(u, v) du dv

and after that minimize f(a). All this is straightforward with SymPy, it remains only to compute p(u, v).

Even though the speed of each car is sampled uniformly from [1, 2] in the "zoomed out" statement, when we look at a
particular intersection the speed of each car is not uniformly distributed, since then the distribution is "conditional
on an intersection taking place". Intuitively, more intersections will take place between cars with speeds that differ
a lot than between cars with similar speeds.

Say that the car u spawns at (x, t) = (0, 0). Draw the region (x, t) where v could spawn such that v would overtake u,
assuming trips of length N. This region is a parallelogram with vertices
    * (0, 0)
    * (N, N/u)
    * (0, N(v - u)/(uv))
    * (-N, N/v)

Then,
    p(u, v) ~ Area of parallelogram
            = 2 * (Area of triangle which is the portion of the parallelogram with x >= 0)
            = base * height
            = [ N(v - u)/(uv) ] * N
            ~ (v - u)/(uv)
where "~" means "proportional to". Since we only care about optimization, we can ignore the constant factor.
The remaining computations can be done with SymPy.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sympy import diff, integrate
from sympy.abc import a, u, v

p = (v - u) / (u * v)
I_slow = integrate(u**2 * p, (v, u, a), (u, 1, a)).simplify()
I_fast = (integrate((u - a) ** 2 * p, (v, u, 2), (u, a, 2))).simplify()
f = (I_fast + I_slow).simplify()
df = diff(f, a)


def fun(x: NDArray) -> float:
    x_ = x.item()
    f_x = f.subs(a, x_)
    return float(f_x)


def jac(x: NDArray) -> NDArray:
    x_ = x.item()
    df_x = df.subs(a, x_)
    return np.array([float(df_x)])


ans = minimize(fun, [1.5], jac=jac, tol=1e-20).x.item()
print(f"{ans = :.10f}")
