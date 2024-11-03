"""
Let E be the event that the circle is contained in the square. We wish
to find ans = 1 - P(E). Denote by C = (X, Y) the center of the circle (a
random variable).

Imagine the square as having side=2 and center at (0, 0). Then,

P(E) = ∫_{-1}^1 ∫_{-1}^1 P(E|X=x, Y=y) p_{X,Y}(x, y) dx dy
     = 8 * ∫_0^1 ∫_0^x P(E|X=x, Y=y) p_{X,Y}(x, y) dy dx,

where in the first equality we used the law of the total probability and
in the second one we used the symmetries in the problem. From now on, we
assume that 0 < y < x < 1.

If X=x and Y=y, the set of possible points that the blue/red sampled
points could be is a rectangle with center (x, y) and
    Lx = 2*(1 - x) [width]
    Ly = 2*(1 - y) [height]
This is true because for a fixed center, the blue point determines the
red point and vice-versa.

The set of points that would lead to the circle being inside the square
is a circle with center (x, y) and radius r = Lx / 2.
"""

from sympy import integrate, pi
from sympy.abc import x, y

Lx = 2 * (1 - x)  # width of rectangle
Ly = 2 * (1 - y)  # heigth of rectangle
r = Lx / 2  # radius of circle

P_X = Lx / 2  # p.d.f. of X
P_Y = Ly / 2  # p.d.f. of Y
P_XY = P_X * P_Y  # p_{X,Y}(x, y) = joint p.d.f. of (X, Y) = C

A_circ = pi * r**2  # area of circle
A_rect = Lx * Ly  # area of rectangle
P_EXY = A_circ / A_rect  # P(E|X=x, Y=y)

P_E = 8 * integrate(P_EXY * P_XY, (y, 0, x), (x, 0, 1))
ans = 1 - P_E
print(f"ans = {ans} = {float(ans):.4f}")
# ans = 1 - pi/6 = 0.4764
