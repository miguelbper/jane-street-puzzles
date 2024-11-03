"""Assume that the large circle has radius 1. Each circle in the ring has
radius r = 1/3. Let a = maximum area we can cover p = maximum proportion we can
cover = a / pi.

Then,
    a = 6 * pi * r**2 + x**2 * a,
where x is the radius of a large inner circle containing a smaller ring
of circles.

It remains to compute x. For this, let
    z = distance from the center of the large circle to a circle in the
        smaller ring, along a tangent line to the circle in the smaller
        ring
    y = 1/3 - z

Then, using trigonometry, we have the following equations
    3*(y + z) = 1   (1)               [definition of y]
    (2 * x)**2 = x**2 + 9*z**2        [Pythagorean theorem]
    (1 + x)**2 = (1 + 3*y)**2 + x**2  [Pythagorean theorem]
"""

from sympy import Rational, pi, solve
from sympy.abc import a, x, y, z

# Find x = radius of large inner circle
# ----------------------------------------------------------------------
eqs_xyz = [
    3 * (y + z) - 1,
    (2 * x) ** 2 - x**2 - 9 * z**2,
    (1 + x) ** 2 - (1 + 3 * y) ** 2 - x**2,
]

solutions = solve(eqs_xyz, [x, y, z])
valid_solutions = [s for s in solutions if all(v.is_positive for v in s)]
x, _, _ = valid_solutions[0]
print(f"{x = }")
# x = -2*sqrt(1 + sqrt(3))/3 + 1/3 + 2*sqrt(3)/3


# Find p = max proportion of the area of circle we can cover
# ----------------------------------------------------------------------

# Find area
r = Rational(1, 3)
eq_a = a - (6 * pi * r**2 + x**2 * a)
a = solve(eq_a, a)[0]
print(f"{a = }")
# a = 6*pi/(9 - (-2*sqrt(1 + sqrt(3)) + 1 + 2*sqrt(3))**2)

# Find proportion
p = a / pi
print(f"{p = }", f"\n  = {p.evalf():.6f}")
# p = 6/(9 - (-2*sqrt(1 + sqrt(3)) + 1 + 2*sqrt(3))**2) = 0.783464
