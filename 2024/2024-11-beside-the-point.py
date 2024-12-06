"""Consider the square S = [0, 1]^2.

Let A be the event that after sampling B and R, there exists a point C on the side closest to B
such that d(B, C) = d(R, C).

Let A_{x,y} be the event that for fixed B = (x, y), after sampling R, there exists a point C on the
side closest to B such that d(B, C) = d(R, C).

By symmetry, we may assume that 0 < y < x < 1/2. By the law of total probability, we have

    P(A) = 8 * ∫_0^{1/2} ∫_0^x P(A_{x,y}) dy dx.

It remains to compute P(A_{x,y}), which can be computed geometrically. Draw a region of the square
[0, 1]^2 where R can be such that the desired condition holds. This region is bounded by two
circles C0, C1 with centers at (0, 0) and (0, 1), and each circle passes through B = (x, y). It is
possible to see that

    P(A_{x,y}) = Area(C0 ∩ S) + Area(C1 ∩ S) - 2 * Area(C0 ∩ C1 ∩ S).
"""

from sympy import Rational, atan, integrate, pi
from sympy.abc import x, y

us = [x, 1 - x]
radii_sq = [u**2 + y**2 for u in us]
angles = [atan(y / u) for u in us]
area_slices = [(r_sq * angle - u * y) / 2 for u, r_sq, angle in zip(us, radii_sq, angles)]
area_circles = [(pi * r_sq) / 4 for r_sq in radii_sq]
p_xy = sum(area_circles) - 2 * sum(area_slices)

prob = (8 * integrate(p_xy, (y, 0, x), (x, 0, Rational(1, 2)))).simplify()
print(f"prob = {prob} \n     = {prob.evalf():10f}")
# prob = -log(2)/6 + 1/12 + pi/6
#      = 0.491407578838308
