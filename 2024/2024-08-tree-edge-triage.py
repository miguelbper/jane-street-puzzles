from sympy import solve, diff, Rational
from sympy.abc import a, q
from sympy.core.expr import Expr
import numpy as np
import matplotlib.pyplot as plt


'''
For each p, let Ω be the probability space where each element is an 
infinite complete binary tree, with edges labeled (independently) A with
probability p and B with probability 1 - p.

Define the following events:
    A = "Tree is s.t. A would win if A is the starting player"
    B = "Tree is s.t. A would win if B is the starting player"

Let
    a = P(A)
    b = P(B)

We use the law of total probability to compute P(A). Consider the table:

E   | P(E)    | P(A|E) | Explanation 
----|---------|--------|--------------------------------------------
A,A | p^2     | b(2-b) | (below)
A,B | p(1-p)  | b      | A chooses edge labeled A, now it's B's turn
B,A | p(1-p)  | b      | A chooses edge labeled A, now it's B's turn
B,B | (1-p)^2 | 0      | A has to choose edge labeled B

Here, the event E = {A,A} corresponds to "The root of the tree has edges
labeled A,A", and analogously for E = {A,B}, {B,A}, {B,B}. To see why
P(A|A,A) = b(2 - b), consider the events 
    L = "A wins in left subtree, starting player is B"
    R = "A wins in right subtree, starting player is B"
Then,
    P(A|E) = P(L or R)                   
           = 1 - P((not L) and (not R))
           = 1 - P(not L) P(not R)
           = 1 - (1 - b)^2
           = b(2 - b),
where the first equality follows from the fact that if A wins in either
left/right subtree, then A chooses the corresponding edge.

By the law of total probability, we conclude that 
    a = p^2 b (2 - b) + 2 b p (1 - p)
      = p b (2 - p b)

Analogously, we can compute P(B):
E   | P(E)    | P(A|E) | Explanation 
----|---------|--------|--------------------------------------------
A,A | p^2     | a^2    | A needs to win on left and right subtree
A,B | p(1-p)  | 0      | B chooses any edge labeled B
B,A | p(1-p)  | 0      | B chooses any edge labeled B
B,B | (1-p)^2 | 0      | B chooses any edge labeled B

By the law of total probability, we conclude that 
    b = p^2 a^2
    a = p^3 a^2 (2 - p^3 a^2)

The solution to this problem is the smallest p such that there is a 
solution (p, a) of a = p^3 a^2 (2 - p^3 a^2) with a > 0. To find this
solution, let 
    q := p^3
    0  = q a^2 (2 - q a^2) - a

'''
eqn = q * a**2 * (2 - q * a**2) - a
print(f'{eqn} = 0')

q_ = solve(eqn, [q])[0]
print(f'=>  q(a) = {q_}')

dq = diff(q_, a)
print(f'=> q\'(a) = {dq}')

a0 = solve(dq, a)[0]
print('\nq\'(a) = 0')
print(f'=> a = {a0}')

q0 = q_.subs({a: a0})
print(f'=> q = {q0}')

p0 = q0 ** (Rational(1, 3))
print(f'=> p = {p0}')

'''
a**2*q*(-a**2*q + 2) - a = 0
=>  q(a) = (1 - sqrt(1 - a))/a**2
=> q'(a) = 1/(2*a**2*sqrt(1 - a)) - 2*(1 - sqrt(1 - a))/a**3

q'(a) = 0
=> a = 8/9
=> q = 27/32
=> p = 3*2**(1/3)/4
'''

def plot_graph(q: Expr, a0: Expr, p0: Expr, ticks: int) -> None:

    p = lambda x: q.subs({a: x}) ** (1/3)
    xs = np.linspace(0, 1, ticks)
    ys = np.array([p(x) for x in xs])

    plt.plot(xs, ys, label='p(a)')
    plt.scatter(float(a0), float(p0), color='red', label=f'{a0 = }, {p0 = }')

    plt.xlabel('a')
    plt.ylabel('p(a)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    
    plt.show()


plot_graph(q_, a0, p0, 1000)
