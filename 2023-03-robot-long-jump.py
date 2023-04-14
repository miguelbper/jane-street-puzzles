from sympy import exp, diff, integrate, nsolve
from sympy.abc import s, l, m

'''
Blog post with detailed explanation of this solution:
https://miguelbper.github.io/2023/04/04/js-2023-03-robot-long-jump.html

For a given robot, there is a number λ such that if the robot lands at x
after the random [0, 1] draw, then:
- if x < λ, then the robot decides to wait;
- if x >= λ, then the robot decides to jump.
Therefore, the number λ fully characterizes a robot's strategy.

Define
    S_λ := score of a robot using strategy λ (this is a random variable)
    F_λ := cumulative distribution function of S_λ
    f_λ := probability density function of S_λ

We start by computing F_λ(s) = P(S_λ <= s). For this, define
    p_{λ,s}(x)   := P(S_λ <= s | robot starts at x)
    p_{λ,s}(x|y) := P(S_λ <= s | robot starts at x, random draw is y)
    f_U          := p.d.f. of uniform distribution

Notice that F_λ(s) = p_{λ,s}(0). By the law of total probability,

    p_{λ,s}(x)
    = ∫_0^1 p_{λ,s}(x|y) f_U(y) dy
    = ∫_0^1 p_{λ,s}(x|y) dy
    = ∫_{0}^{λ-x} p_{λ,s}(x|y) dy 
      + ∫_{λ-x}^{1-x} p_{λ,s}(x|y) dy 
      + ∫_{1-x}^{1} p_{λ,s}(x|y) dy
    = ∫_{0}^{λ-x} p_{λ,s}(x+y) dy 
      + ∫_{λ-x}^{1-x} len([0,s] ∩ [x+y, x+y+1]) dy 
      + ∫_{1-x}^{1} 1 dy
    = ∫_{x}^{λ} p_{λ,s}(u) du 
      + ∫_{λ}^{1} len([0,s] ∩ [u, u+1]) du 
      + x

Define h_{λ,s} = ∫_{λ}^{1} len([0,s] ∩ [u, u+1]) du. Then, we have shown
    p_{λ,s}(x) = ∫_{x}^{λ} p_{λ,s}(u) du + h_{λ,s} + x.

This implies that:
    p_{λ,s}(λ) = h_{λ,s} + λ,
    d/dx p_{λ,s}(x) = - p_{λ,s}(x) + 1.

This is a 1st order, linear, constant coefficient ODE. We are also given
one boundary condition. The solution of the boundary value problem is
    p_{λ,s}(x) = 1 + (h_{λ,s} + λ - 1) exp(λ - x).

Therefore, F_λ(s) = 1 + (h_{λ,s} + λ - 1) exp(λ).

It is possible to show that h_{λ,s} is given by:
    h0 = 0                                                    if s<λ
    h1 = (s - λ)**2 / 2                                       if λ<s<1
    h2 = 1/2 (1 - λ)(2s - λ - 1)                              if 1<s<1+λ
    h3 = (s - 1 - λ)(2 - s) + (2 - s)^2/2 + (1 - λ)(s - 1)    if 1+λ<s<2 
'''

h0 = 0                               
h1 = (s - l)**2 / 2                      
h2 = (1 - l) * (2*s - l - 1) / 2
h3 = (s - 1 - l)*(2 - s) + (2 - s)**2/2 + (1 - l)*(s - 1)

# Cumulative distribution function
F0 = 1 + (h0 + l - 1) * exp(l)
F1 = 1 + (h1 + l - 1) * exp(l)
F2 = 1 + (h2 + l - 1) * exp(l)
F3 = 1 + (h3 + l - 1) * exp(l)

# Probability density function
f0 = diff(F0, s)
f1 = diff(F1, s)
f2 = diff(F2, s)
f3 = diff(F3, s)


'''
Now we wish to compute the optimal strategy for each robot. Define
    p(λ, μ) := P(R1 wins | R1 plays λ and R2 plays μ).

By the law of total probability,
    p(λ, μ)
    
    =   p(λ, μ | S_λ=0 and S_μ=0) P(S_λ=0 and S_μ=0)
      + p(λ, μ | S_λ=0 and S_μ>0) P(S_λ=0 and S_μ>0)
      + p(λ, μ | S_λ>0 and S_μ=0) P(S_λ>0 and S_μ=0)
      + p(λ, μ | S_λ>0 and S_μ>0) P(S_λ>0 and S_μ>0)
    
    =   p(λ, μ) P(S_λ=0) P(S_μ=0)
      +       0 P(S_λ=0) P(S_μ>0)
      +       1 P(S_λ>0) P(S_μ=0)
      + P(R1 wins and S_λ>0 and S_μ>0)

    = p(λ, μ) P(S_λ=0) P(S_μ=0) + P(S_λ>0) P(S_μ=0) + P(S_λ > S_μ > 0)

which implies that
    p(λ, μ) = (P(S_λ>0) P(S_μ=0) + P(S_λ > S_μ > 0)) 
              / (1 - P(S_λ=0) P(S_μ=0)).

We need to compute q(λ, μ) := P(S_λ > S_μ > 0). It will be enough to 
know p(λ, μ) and q(λ, μ) in the case λ < μ.

    q(λ, μ)
    = P(S_λ > S_μ > 0)
    = ∫_{λ}^{2} ∫_{μ}^{x} f_λ(x) f_μ(y) dy dx
    = ∫_{λ}^{2} f_λ(x) ∫_{μ}^{x} f_μ(y) dy dx
    = ∫_{λ}^{2} f_λ(x) (F_μ(x) - F_μ(μ)) dx
    
    =   ∫_{  λ}^{  μ} f_λ(x) (F_μ(x) - F_μ(μ)) dx
      + ∫_{  μ}^{  1} f_λ(x) (F_μ(x) - F_μ(μ)) dx
      + ∫_{  1}^{λ+1} f_λ(x) (F_μ(x) - F_μ(μ)) dx
      + ∫_{λ+1}^{μ+1} f_λ(x) (F_μ(x) - F_μ(μ)) dx
      + ∫_{μ+1}^{  2} f_λ(x) (F_μ(x) - F_μ(μ)) dx

    =   ∫_{  λ}^{  μ} f1_λ(x) (F0_μ(x) - F0_μ(μ)) dx
      + ∫_{  μ}^{  1} f1_λ(x) (F1_μ(x) - F0_μ(μ)) dx
      + ∫_{  1}^{λ+1} f2_λ(x) (F2_μ(x) - F0_μ(μ)) dx
      + ∫_{λ+1}^{μ+1} f3_λ(x) (F2_μ(x) - F0_μ(μ)) dx
      + ∫_{μ+1}^{  2} f3_λ(x) (F3_μ(x) - F0_μ(μ)) dx
'''
I0 = integrate(f1 * (F0.subs({l:m}) - F0.subs({l:m, s:m})), (s,   l,   m))
I1 = integrate(f1 * (F1.subs({l:m}) - F0.subs({l:m, s:m})), (s,   m,   1))
I2 = integrate(f2 * (F2.subs({l:m}) - F0.subs({l:m, s:m})), (s,   1, l+1))
I3 = integrate(f3 * (F2.subs({l:m}) - F0.subs({l:m, s:m})), (s, l+1, m+1))
I4 = integrate(f3 * (F3.subs({l:m}) - F0.subs({l:m, s:m})), (s, m+1,   2))
q = I0 + I1 + I2 + I3 + I4 # type: ignore
p = ((1 - F0)*(F0.subs({l:m})) + q) / (1 - F0 * F0.subs({l:m})) 


'''
We now compute the Nash equilibrium of this game, as well as the answer
to the original question.

Fact: The optimal strategies (λ0, μ0) are such that ∇p(λ0, μ0) = 0.
Explanation: If ∇p(λ0, μ0) != 0, it would be possible for one of the 
robots to change their strategy and obtain a higher probability of 
winning.

Fact: The optimal strategies (λ0, μ0) are such that λ0 = μ0.
Explanation: the game is symmetric.

Fact: The optimal strategy satisfies (∂/∂λ p)(λ0, λ0) = 0.
Explanation: by the two facts above.

We can solve (∂/∂λ p)(λ0, λ0) = 0 to find λ0. The answer to the puzzle 
is F0_λ0(0).
'''
dp = diff(p, l).subs({m:l})
l0 = nsolve(dp, 0.5, prec=50)
ans = F0.subs({l: l0})
print(f'λ0                = {l0:.9f}')
print(f'P(Robot scores 0) = {ans:.9f}')
# λ0                = 0.416195355
# P(Robot scores 0) = 0.114845886