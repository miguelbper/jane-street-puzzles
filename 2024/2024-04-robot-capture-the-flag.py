"""Let x be the radius that Erin uses. Define.

    f(x) := P(Aaron wins | Erin uses radius x)

In equilibrium, Erin will choose the radius that minimizes the
probability that Aaron wins. Therefore, the answer to the puzzle is

    ans := min_{x in [0, 1]} f(x)

Let
    - P(A|x) = P(Aaron wins | Erin uses radius x)
    - P(A|x,r) = P(Aaron wins | Erin uses radius x, Flag is at radius r)
    - P(r) = P(Flag is at r) = 2r

Then,
    f(x) = P(A|x)                    [by definition]
         = ∫_0^1 P(A|x,r) P(r) dr    [law of total probability]

To compute P(r), notice that the flag is uniformly distributed in the
unit circle, which implies that P(r) is proportional to r: P(r) = Cr.
This, combined with the fact that ∫_0^1 P(r) dr = 1, implies that
P(r) = 2r.

To compute P(A|x,r), we imagine that Aaron knows that Erin is using
radius x and that the Flag is at radius r. The optimal strategy for
Aaron in this scenario is to choose a random angle β, choose the optimal
radius y = y(x, r), and to move to the point y exp(iβ). There are two
cases
    - r < 2/x: Aaron can choose y = 0 and win with probability 1.
    - r > 2/x: Imagine a circle with center the Flag and radius |x-r|.
        Then, Erin is in the boundary of this circle. Aaron wins if he
        lands inside this circle. The y that maximizes the probability
        of this happening is the one that maximizes the angle interval
        [β_0, β_1] such that the path y exp(iβ) for β in [β_0, β_1]
        is inside the circle. This happens when the lines emanating from
        the origin at an angle of β_0 and β_1 are tangent to the circle.
        Using trigonometry, this implies that

            y = sqrt(x(2r-x))
            P(A|x,r) = acos(y/r)/π = acos(sqrt(x(2r-x))/r)/π

Using Wolfram Mathematica, we can show that

    f(x) = ∫_0^1 P(A|x,r) P(r) dr
         = ((32 + 3*pi)*x**2 - 8*(1 + x)*s + 24*acos(s)) / (24*pi),

where s := sqrt(x*(2 - x)).
"""

from sympy import Symbol, acos, diff, nsolve, pi, sqrt

x = Symbol("x", positive=True)

# compute f(x) according to the formula above
s = sqrt(x * (2 - x))
f = ((32 + 3 * pi) * x**2 - 8 * (1 + x) * s + 24 * acos(s)) / (24 * pi)

# compute the minimum of f(x)
df = diff(f, x)
x0 = nsolve(df, 0.5, prec=50)
ans = f.subs(x, x0)
print(f"x0          = {x0:.10f}")
print(f"ans = f(x0) = {ans:.10f}")
"""X0          = 0.5013069942 ans = f(x0) = 0.1661864865."""
