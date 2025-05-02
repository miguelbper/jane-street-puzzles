"""Let 0 <= p <= 1.

Define:

A := Event that there exists an infinite path down the tree that sums to at most 1
B := Event that there exists an infinite path down the tree that sums to at most 0
R := Random variable that is the value of the root of the tree

a := P(A)
b := P(B)

We want to compute f(p) := P(A)

| r | P(R = r) | P(A | R = r)                                                          |
|---|----------|-----------------------------------------------------------------------|
| 0 | p        | P(A_L or A_R) = 1 - P(not(A_L or A_R)) = 1 - (1 - P(A))**2 = a(2 - a) |
| 1 | 1 - p    | P(B_L or B_R) = 1 - P(not(B_L or B_R)) = 1 - (1 - P(B))**2 = b(2 - b) |

a = p * a(2 - a) + (1 - p) * b(2 - b)

| r | P(R = r) | P(B | R = r)                                                          |
|---|----------|-----------------------------------------------------------------------|
| 0 | p        | P(B_L or B_R) = 1 - P(not(B_L or B_R)) = 1 - (1 - P(B))**2 = b(2 - b) |
| 1 | 1 - p    | 0                                                                     |

b = p * b(2 - b)

It only remains to solve the system of equations

    a = 1/2
    a = p * a(2 - a) + (1 - p) * b(2 - b)
    b = p * b(2 - b)
"""

import matplotlib.pyplot as plt
import numpy as np
from sympy import Eq, Rational, solve
from sympy.abc import a, b, p

eq_b = Eq(b, p * b * (2 - b))
eq_a = Eq(a, p * a * (2 - a) + (1 - p) * b * (2 - b))

b_sol = solve(eq_b, b)[1]  # Chose branch with p >= 1/2 and 0 <= b <= 1
a_sol = solve(eq_a.subs(b, b_sol), a)[1]  # Chose branch with p >= 1/2 and 0 <= a <= 1

eq = Eq(Rational(1, 2), a_sol)
p_sol = solve(eq, p)[-1]
p_val = float(p_sol.evalf())
print(f"{a_sol = }")
print(f"{b_sol = }")
print(f"{p_sol = }")
print(f"answer = {p_val:.10f}")
# a_sol = (p**2*(2*p - 1) + sqrt(p**3*(2*p - 1)*(2*p**2 - 5*p + 4)))/(2*p**3)
# b_sol = 2 - 1/p
# p_sol = -(134/27 + 2*sqrt(57)/3)**(1/3)/3 + 8/(27*(134/27 + 2*sqrt(57)/3)**(1/3)) + 10/9
# answer = 0.5306035754


# Plot the solution
P = np.linspace(0, 1, 1000)
A = [0 if p_val <= 0.5 else a_sol.subs(p, p_val) for p_val in P]
B = [0 if p_val <= 0.5 else b_sol.subs(p, p_val) for p_val in P]

plt.figure(figsize=(10, 10))
plt.plot(P, A, "r-", label="P(A)")
plt.plot(P, B, "b-", label="P(B)")
plt.plot(p_val, 0.5, "ko", label="Solution")
plt.axhline(y=0.5, color="k", linestyle="--", alpha=0.5)
plt.axvline(x=p_val, color="k", linestyle="--", alpha=0.5)
plt.xticks([0, p_val, 1], ["0", f"{p_val:.10f}", "1"])
plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
plt.grid(True)
plt.gca().set_aspect("equal")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("p")
plt.ylabel("P(*)")
plt.title("P(A) and P(B) as functions of p")
plt.legend()
plt.show()
