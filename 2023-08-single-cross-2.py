import math
from random import uniform
import matplotlib.pyplot as plt
import numpy as np
from sympy import integrate, symbols, cos, sin, pi, simplify, diff, solveset


# Closed-form solution with SymPy
# ----------------------------------------------------------------------
'''
Blog post with detailed explanation of this solution:
https://miguelbper.github.io/2023/09/05/js-2023-08-single-cross-2.html

Consider the lattice Z^3 in R^3.
For a parameter D, consider the experiment of choosing x, y, z in [0,1],
and a point in the sphere or radius D uniformly at random. Consider the 
line segment with endpoints (x, y), and the chosen point in the sphere. 
Define random variables
    N  = num times the segment crosses a side of a cube
    Nx = num times the segment crosses a side of a cube along the x axis
    Ny = num times the segment crosses a side of a cube along the y axis
    Nz = num times the segment crosses a side of a cube along the z axis

We wish to compute P(N=1) for each D, and find D which maximizes P(N=1).

Fact: We may assume D < sqrt(3).
Explanation: If D = sqrt(3) then P(N>=1) = 1. Also, P(N>=2) is 
nondecreasing in D for D >= sqrt(3).

Consider spherical coordinates theta in [0, 2pi] and phi in [0, pi].

Fact: We may assume that theta in [0, pi/4] and phi in [0, pi/2].
Explanation: by symmetry, the two possibilities give the same 
distribution for N.


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


A Monte Carlo simulation (done below) reveals that the optimal value of D
occurs for D < 1. So, it suffices to compute P(N=1) for D<=1. Let
    Lx = D cos(θ) sin(𝜑)
    Ly = D sin(θ) sin(𝜑)
    Lz = D cos(𝜑)

We compute P(N=1):

P(N=1) = P(Nx=1, Ny=0, Nz=0) + P(Nx=0, Ny=1, Nz=0) + P(Nx=0, Ny=0, Nz=1)
       
       =   (4/π) * ∫_0^{π/4} ∫_0^{π/2} P(Nx=1, Ny=0, Nz=0 | θ,𝜑) sin(𝜑) d𝜑 dθ
         + (4/π) * ∫_0^{π/4} ∫_0^{π/2} P(Nx=0, Ny=1, Nz=0 | θ,𝜑) sin(𝜑) d𝜑 dθ
         + (4/π) * ∫_0^{π/4} ∫_0^{π/2} P(Nx=0, Ny=0, Nz=1 | θ,𝜑) sin(𝜑) d𝜑 dθ
       
       =   (4/π) * ∫_0^{π/4} ∫_0^{π/2} P(Nx=1|θ,𝜑) P(Ny=0|θ,𝜑) P(Nz=0|θ,𝜑) sin(𝜑) d𝜑 dθ
         + (4/π) * ∫_0^{π/4} ∫_0^{π/2} P(Nx=0|θ,𝜑) P(Ny=1|θ,𝜑) P(Nz=0|θ,𝜑) sin(𝜑) d𝜑 dθ
         + (4/π) * ∫_0^{π/4} ∫_0^{π/2} P(Nx=0|θ,𝜑) P(Ny=0|θ,𝜑) P(Nz=1|θ,𝜑) sin(𝜑) d𝜑 dθ

       =   (4/π) * ∫_0^{π/4} ∫_0^{π/2} Lx (1-Ly) (1-Lz) sin(𝜑) d𝜑 dθ
         + (4/π) * ∫_0^{π/4} ∫_0^{π/2} (1-Lx) Ly (1-Lz) sin(𝜑) d𝜑 dθ
         + (4/π) * ∫_0^{π/4} ∫_0^{π/2} (1-Lx) (1-Ly) Lz sin(𝜑) d𝜑 dθ
       
Here, in the first equality we used N = Nx + Ny + Nz, in the second we 
used the law of total probability, in the third we used the fact that 
for fixed θ,𝜑, the events {Nx=i, Ny=j, Nz=k} are independent, and in the
fourth we used the computation in R^1 from above (with D <= 1).
'''

d, th, ph = symbols('d th ph')

lx = d * cos(th) * sin(ph)
ly = d * sin(th) * sin(ph)
lz = d * cos(ph)

IX = lx * (1 - ly) * (1 - lz) 
IY = (1 - lx) * ly * (1 - lz)
IZ = (1 - lx) * (1 - ly) * lz
I = IX + IY + IZ

p = (4/pi) * integrate(integrate(I * sin(ph), (ph, 0, pi/2)), (th, 0, pi/4))
p = p.simplify()
print(f'p(d)  = {p}')
# p(d)  = d*(3*d**2 - 16*d + 6*pi)/(4*pi)

pdv = diff(p, d).simplify()
print(f"p'(d) = {pdv}")
# p'(d) = (9*d**2 - 32*d + 6*pi)/(4*pi)

d0 = [sol for sol in solveset(pdv, d) if 0 <= sol <= 1][0]
print(f"p'(d) = 0 => d0 = {d0} = {d0.evalf():.10f}")
# p'(d) = 0 => d0 = -sqrt(2)*sqrt(128 - 27*pi)/9 + 16/9 = 0.7452572091

p0 = p.subs({d: d0}).simplify()
print(f'p(d0) = {p0} = {p0.evalf():.10f}')
print(f'ans   = ({d0.evalf():.10f}, {p0.evalf():.10f})')
# p(d0) = -2048/(243*pi) - sqrt(256 - 54*pi)/9 + 128*sqrt(256 - 54*pi)/(243*pi) + 8/3 
#       = 0.5095346021
# ans   = (0.7452572091, 0.5095346021)


# Monte Carlo Simulation
# ----------------------------------------------------------------------
NUM_EXPERIMENTS = 10**4


def num_crossings_1d(l: float, x: float) -> int:
    if l < 0:
        l = -l
        x = 1 - x
    return math.floor(x + l)


def num_crossings(
    d_: float, 
    x: float, 
    y: float,
    z: float, 
    theta: float,
    phi: float,
) -> int:
    lx = d_ * math.cos(theta) * math.sin(phi)
    ly = d_ * math.sin(theta) * math.sin(phi)
    lz = d_ * math.cos(phi)
    nx = num_crossings_1d(lx, x)
    ny = num_crossings_1d(ly, y)
    nz = num_crossings_1d(lz, z)
    return nx + ny + nz


def prob_one_crossing_mc(d_: float) -> float:
    count = 0

    for _ in range(NUM_EXPERIMENTS):
        # throw segment unif random
        x = uniform(0, 1)
        y = uniform(0, 1)
        z = uniform(0, 1)

        # see https://mathworld.wolfram.com/SpherePointPicking.html
        u = uniform(0, 1)
        v = uniform(0, 1)
        theta = 2 * math.pi * u
        phi = math.acos(2*v - 1)

        # compute num_crossings
        count += 1 == num_crossings(d_, x, y, z, theta, phi)
        
    return count / NUM_EXPERIMENTS


# Plots
# ----------------------------------------------------------------------
NUM_POINTS = 100
EPS = 1/10

d0_ = float(d0.evalf())
p0_ = float(p0.evalf())

ds_mc = np.linspace(0, math.sqrt(3), NUM_POINTS)
ps_mc = np.array([prob_one_crossing_mc(a) for a in ds_mc])
plt.plot(ds_mc, ps_mc, label='monte carlo')

ds_cl = np.linspace(0, 1, NUM_POINTS)
ps_cl = np.array([p.subs({d: a}) for a in ds_cl])
plt.plot(ds_cl, ps_cl, label='closed form')

plt.scatter(d0_, p0_, color='red')

plt.xlabel('D')
plt.ylabel('P(N=1)')
plt.legend()
plt.grid(True)
plt.show()