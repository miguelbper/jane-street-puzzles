import math
from random import uniform
import matplotlib.pyplot as plt
import numpy as np
from sympy import integrate, symbols, cos, sin, pi, acos


# Closed-form solution with SymPy
# ----------------------------------------------------------------------

'''
Consider the lattice Z^2 in R^2.
For a parameter D, consider the experiment of choosing x, y in [0, 1],
theta in [0, 2 pi] uniformly at random. Consider the line segment with
one endpoint (x, y), angle theta and length D. Define random variables
    N  = num times the segment crosses a side of a square
    Nx = num times the segment crosses a vertical side of a square
    Ny = num times the segment crosses a horizontal side of a square

We wish to compute P(N=1) for each D, and find D which maximizes P(N=1).

Fact: We may assume D < sqrt(2).
Explanation: If D = sqrt(2) then P(N>=1) = 1. Also, P(N>=2) is
nondecreasing in D for D >= sqrt(2).

Fact: We may assume that theta in [0, pi/4].
Explanation: by symmetry, the two experiments (theta in [0, pi/4] and
theta in [0, 2 pi]) give the same distribution for N.


We now solve this problem in R^1 (we will later reduce to this case).
For a parameter L (length of a segment), choose x in [0, 1] uniformly at
random. Define
    N = num crossings
      = # {n in Z | x < n <= x + L}
      = # {1,...,floor(x + L)}
      = floor(x + L)
Let delta = L - floor(L). The previous computation implies that the
distribution of N is
    P(N = floor(L))     = 1 - delta
    P(N = floor(L) + 1) = delta


We compute P(N=1):
P(N=1) = P(Nx=1, Ny=0) + P(Nx=0, Ny=1)
       = (4/π) * ∫_0^{π/4} (P(Nx=1, Ny=0|θ) + P(Nx=0, Ny=1|θ)) dθ
       = (4/π) * ∫_0^{π/4} (P(Nx=1|θ) P(Ny=0|θ) + P(Nx=0|θ) P(Ny=1|θ)) dθ
Here, in the first equality we used N = Nx + Ny, in the second we used
the law of total probability, and in the third we used the fact that for
a fixed theta, Nx=i and Ny=j are independent.

We now compute P(Nx=i|θ) and P(Ny=j|θ) using the previous 1d result. For
this, let
    Lx = D cos(θ)
    Ly = D sin(θ)


Case 1: D <= 1
P(N=1) = (4/π) * ∫_0^{π/4} (P(Nx=1|θ) P(Ny=0|θ) + P(Nx=0|θ) P(Ny=1|θ)) dθ
       = (4/π) * ∫_0^{π/4} (Lx (1 - Ly) + (1 - Lx) Ly) dθ


Case 2: 1 <= D <= sqrt(2)
Define θ_D = arccos(1/D).

P(N=1) = (4/π) * ∫_0^{π/4} (P(Nx=1|θ) P(Ny=0|θ) + P(Nx=0|θ) P(Ny=1|θ)) dθ
       = (4/π) * ∫_0^{θ_D} (P(Nx=1|θ) P(Ny=0|θ) + P(Nx=0|θ) P(Ny=1|θ)) dθ
         + (4/π) * ∫_{θ_D}^{π/4} (P(Nx=1|θ) P(Ny=0|θ) + P(Nx=0|θ) P(Ny=1|θ)) dθ
       = (4/π) * ∫_0^{θ_D} ((2 - Lx) (1 - Ly) + 0 Ly) dθ
         + (4/π) * ∫_{θ_D}^{π/4} (Lx (1 - Ly) + (1 - Lx) Ly) dθ
'''

d, th = symbols('d th')

lx = d * cos(th)
ly = d * sin(th)

p1 = (4/pi) * integrate(lx * (1 - ly) + (1 - lx) * ly, (th, 0, pi/4))
p1 = p1.simplify()

td = acos(1/d)
IA = integrate((2 - lx) * (1 - ly), (th, 0, td))
IB = integrate(lx * (1 - ly) + (1 - lx) * ly, (th, td, pi/4))
p2 = (4/pi) * (IA + IB)
p2 = p2.simplify()

print(f'p1 = {p1}')
print(f'p2 = {p2}')
# p1 = 2*d*(2 - d)/pi
# p2 = 2*(2*d**2 - 4*d*sqrt(1 - 1/d**2) - 4*d + 4*acos(1/d) + 3)/pi


def prob_one_crossing_cl(d_: float) -> float:
    return p1.subs({d: d_}) if d_ <= 1 else p2.subs({d: d_})


# The plot below shows that P(N=1) attains its max when d = 1.
p_max = p1.subs({d: 1})
p_max_fl = float(p_max.evalf())
print(f'p_max = {p_max} = {p_max_fl:.4f}')
# p_max = 2/pi = 0.6366


# Monte Carlo Simulation
# ----------------------------------------------------------------------
NUM_EXPERIMENTS = 10**4


def num_crossings_1d(l: float, x: float) -> int:
    if l < 0:
        l = -l
        x = 1 - x
    return math.floor(x + l)


def num_crossings(d_: float, x: float, y: float, theta: float) -> int:
    lx = d_ * math.cos(theta)
    ly = d_ * math.sin(theta)
    nx = num_crossings_1d(lx, x)
    ny = num_crossings_1d(ly, y)
    return nx + ny


def prob_one_crossing_mc(d_: float) -> float:
    count = 0

    for _ in range(NUM_EXPERIMENTS):
        # throw x, y, theta unif random
        x = uniform(0, 1)
        y = uniform(0, 1)
        theta = uniform(0, 2*pi)

        # compute num_crossings
        count += 1 == num_crossings(d_, x, y, theta)

    return count / NUM_EXPERIMENTS


# Plots
# ----------------------------------------------------------------------
NUM_POINTS = 100
ds = np.linspace(0, math.sqrt(2), NUM_POINTS)

# uncomment the following two lines to run and plot MC simulation
# ps_mc = np.array([prob_one_crossing_mc(d) for d in ds])
# plt.plot(ds, ps_mc, label='monte carlo')

ps_cl = np.array([prob_one_crossing_cl(d) for d in ds])
plt.plot(ds, ps_cl, label='closed form')

plt.scatter(1, p_max_fl, color='red')

plt.xlabel('D')
plt.ylabel('P(N=1)')
plt.legend()
plt.grid(True)
plt.show()
